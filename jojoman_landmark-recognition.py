import tensorflow.compat.v1 as tf

import tensorflow.compat.v1.gfile as gfile

# from tensorflow.python.platform import gfile

import csv,os,json,cv2,math,shutil,time

import numpy as np

import matplotlib.pyplot as plt

import sys



tf.disable_v2_behavior()



root_dir = os.path.join('..', 'input')

model_dir = os.path.join(root_dir,'pretrainedmodel')

data_dir = os.path.join(root_dir,'landmark-recognition-2020')

label_dict_dir = os.path.join(root_dir,'label-dict')



sys.path.append(model_dir)

sys.path.append(root_dir)

import tf_slim as slim

img_format = {'png', 'bmp', 'jpg'}

from inception_resnet_v1 import inference as inception_resnet_v1 



test_root_dir = os.path.join(data_dir,'test')

train_root_dir = os.path.join(data_dir,'train')



csv_path = os.path.join(data_dir,'train.csv')

pb_path = os.path.join(model_dir,'pb_model.pb')

node_dict = {'input': 'input:0',

             'phase_train':'phase_train:0',

             'embeddings': 'embeddings:0',

             'keep_prob': 'keep_prob:0'

             }

img_format = {'png', 'bmp', 'jpg'}

predict_num = 20

data_range = None





#----get train paths

#train_uid_path,label_dict = get_path_labels(train_root_dir,csv_path)
#----from the label dict to get the label2uid dict

label2uid_dict = dict()

json_files = [file.path for file in os.scandir(label_dict_dir) if file.name.split(".")[-1] == 'json']

with open(json_files[0],'r') as f:

    label_dict = json.load(f)

    

for uid, label in label_dict.items():

    label2uid_dict[label] = uid
# print(tf.__version__)



for obj in os.scandir(label_dict_dir):

#     if obj.is_dir():

    print(obj.path)
#----get test paths

test_paths = list()



for dir_name, sub_dirnames, filenames in os.walk(test_root_dir):

    if len(filenames) > 0:

        for file in filenames:

            if file.split(".")[-1] in img_format:

                full_path = os.path.join(dir_name,file)

                test_paths.append(full_path)

len_path = len(test_paths)

print("test image quantity:",len_path)

class Prediction():

    def __init__(self):

        pass

        # ----var

        #paths = list()

        

#         print("--------Prediction init --------")



#         #----get paths from json file

# #         if isinstance(path_source, str):

# #             if path_source[-4:] == 'json':

# #                 with open(path_source, 'r')as f:

# #                     content = json.load(f)

# #                     paths = content['paths']



#         len_path = len(paths)





#         if len_path == 0:

#             raise  ValueError



#         else:

#             print("image quantity:",len_path)

#             #----local var to global

#             self.paths = paths

#             self.len_path = len_path

#             #self.path_source = path_source



    def model_init(self):

        #----var

        model_shape = [None,112,112,3]

        feature_num = 256

        class_num = 81313

        lr = 5e-4

        

        print("--------Prediction model init --------")



        # ----tf placeholder

        tf_input = tf.placeholder(dtype=tf.float32, shape=model_shape, name="input")

        tf_label_batch = tf.placeholder(dtype=tf.int32, shape=[None], name="label_batch")

        tf_phase_train = tf.placeholder(tf.bool, name='phase_train')

        tf_keep_prob = tf.placeholder(tf.float32, name='keep_prob')



        #----inference

        prelogits, _ = inception_resnet_v1(tf_input, tf_keep_prob, phase_train=tf_phase_train,

                                           bottleneck_layer_size=feature_num,

                                           weight_decay=0.0)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')



        #----loss

        logits = tf.layers.dense(inputs=prelogits, units=class_num, activation=None, name="logits")

        predict = tf.nn.softmax(logits, name='predict')

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(

            labels=tf_label_batch, logits=logits), name='loss')



        #----opt

        #optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)



        # ----local vars to global

        self.model_shape = model_shape

        self.feature_num = feature_num

        self.lr = lr

        self.tf_input = tf_input

        self.tf_label_batch = tf_label_batch

        self.tf_phase_train = tf_phase_train

        self.tf_keep_prob = tf_keep_prob

        self.embeddings = embeddings

        self.prelogit = prelogits

        self.predict = predict



    def get_prediction(self,paths,save_dir,label2uid_dict,predict_num=1):

        #----var

        batch_size = 192

        predictions = list()

        saver = tf.train.Saver(max_to_keep=2)

        argsort = list()

        

        #----paths

        len_path = len(paths)

        if len_path == 0:

            raise  ValueError

        else:

            print("image quantity:",len_path)



        #----GPU setting

        config = tf.ConfigProto(log_device_placement=True,

                                allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備

                                )

        config.gpu_options.allow_growth = True



        with tf.Session(config=config) as sess:

            # ----restore the model

            files = [file.name for file in os.scandir(save_dir) if file.name.split(".")[-1] == "meta"]



            if len(files) == 0:  # 沒有任何之前的權重

                sess.run(tf.global_variables_initializer())

                print('no previous model param can be used')

            else:

                # self.saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))

                num_list = list()

                for file in files:

                    num_list.append(int(file.split(".")[0].split("-")[-1]))

                argmax = np.argmax(num_list)

                model_path = os.path.join(model_dir,files[argmax].split(".")[0])

                #check_name = files[-1].split("/")[-1].split(".")[0]

                #model_path = os.path.join(save_dir, check_name)

                saver.restore(sess, model_path)

                msg = 'use previous model param:{}'.format(model_path)

                print(msg)



            feed_dict = {self.tf_phase_train: False, self.tf_keep_prob: 1.0}



            ites = math.ceil(len_path / batch_size)

            for idx in range(ites):

                num_start = idx * batch_size

                num_end = np.minimum(num_start + batch_size, len_path)



                # ----batch data

                batch_dim = [num_end - num_start]

                batch_dim.extend(self.model_shape[1:])

                batch_data = np.zeros(batch_dim, dtype=np.float32)

                for idx_path, path in enumerate(paths[num_start:num_end]):

                    img = cv2.imread(path)

                    if img is None:

                        print("read failed:", path)

                    else:

                        img = cv2.resize(img, (self.model_shape[2], self.model_shape[1]))

                        img = img[:, :, ::-1]

                        batch_data[idx_path] = img



                # ----img data norm

                batch_data /= 255

                print("batch_data shape:", batch_data.shape)



                feed_dict[self.tf_input] = batch_data



                temp_predict = sess.run(self.predict, feed_dict=feed_dict)#[batch_size,81313]

                #print("num_start:{}, num_end:{}, temp_predict shape:{}".format(num_start, num_end, temp_predict.shape))

                temp_argsort = np.argsort(temp_predict,axis=-1)

                temp_argsort = temp_argsort[:, -predict_num:]

                argsort.extend(temp_argsort)

                #----get the biggest predict_num probabilities

                for arg_1, pred in zip(temp_argsort,temp_predict):

                    predictions.append(pred[arg_1])



            #----

            argsort = np.array(argsort)

            # argsort = np.argsort(predictions, axis=-1)

            # argsort = argsort[:, -predict_num:]

            print("argsort shape", argsort.shape)



            #----data distribution

#             prob_list = list()

#             for seq, prob in zip(argsort,predictions):

#                 prob_list.append(prob[seq])



#             argsort.astype(int)

#             prob_list = np.array(prob_list,dtype=float)



            #----output csv

            csv_path = 'submission.csv'

            with open(csv_path, 'w') as submission_csv:

                csv_writer = csv.DictWriter(submission_csv, fieldnames=['id', 'landmarks'])

                csv_writer.writeheader()

                for test_path, label,prediction in zip(paths, argsort,predictions):

                    uid = test_path.split("/")[-1].split(".")[0]

                    

                    score = prediction[-1]

                    csv_writer.writerow({'id': uid, 'landmarks': f'{label2uid_dict[label[-1]]} {score}'})



            print("csv_path is saved in ",csv_path)







#             content = {'paths':self.paths, 'argsort':argsort.tolist(),'prob':prob_list.tolist()}



# #             json_path = os.path.join(os.path.dirname(self.path_source),'argsort_prob.json')

#             json_path = 'argsort_prob.json'

#             with open(json_path, 'w') as f:

#                 json.dump(content,f)

#                 print("json_path is saved in ",json_path)



#             return json_path
#----prediction

'''

●I use my pretrained model weights to do predictions

●I created a classification model with inception resnet v1. The loss func. is the normal cross entropy

●In my original opinions, I wanna use embeddings to do matchings but it needs more than 13 GB RAM.So I gave up.

●Because of data imbalance, specific number images of each class are selected to train. For example, 3 images of each class are randomly selected every epoch to do optimizers without data imbalance.

●Augmentation method is also adopted. But I do it when reading the batch data.

●The most difficult point is not to recognize the landmark but the faces shown in testing images.



'''

predict_num = 1

pre = Prediction()

pre.model_init()

pre.get_prediction(test_paths,model_dir,label2uid_dict,predict_num=predict_num)
