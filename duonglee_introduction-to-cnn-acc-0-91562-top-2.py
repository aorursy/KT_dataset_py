import cv2

import numpy as np

import os



class ReadImage:

    def __init__(self):

        self.name = "Load Image Challenge 1"

        self.unix = np.eye(43)



    def listFileFrom(self, source):

        all, cla = np.array(self.getListOfFile("..\\"+source))

        self.sourceFile = all

        self.sourceClassify = cla

        self.pathLabel = self.folderLable(source)



    def folderLable(self, s):

        arr = []

        for i in range(43):

            arr.append("..\\"+s+"\\"+str(i))

        return arr



    def getListOfFile(self, dirName):

        listOfFile = os.listdir(dirName)

        allFiles = list()

        classify = list()



        for entry in listOfFile:

            fullPath = os.path.join(dirName, entry)

            if os.path.isdir(fullPath):

                all, cla = self.getListOfFile(fullPath)

                allFiles = allFiles + all

                classify = classify + cla

            else:

                allFiles.append(fullPath)

                classify.append(os.path.basename(os.path.dirname(fullPath)))

        return allFiles, classify

    

    #Tăng độ tương phản của ảnh

    def bgr(self, img):

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        lab_planes = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)

        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return bgr

    #Dịch chuyển ảnh đi 5%

    def translation(self, img):

        M = np.float32([[1, 0, 5], [0, 1, 5]])

        dst = cv2.warpAffine(img, M, (48, 48))

        return dst

    

    #Xoay ảnh đi một góc = angle

    def rotation(self, img, angle):

        image_center = tuple(np.array(img.shape[1::-1]) / 2)

        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

        return result

    

    #Đọc ảnh train - f(int) = from, t(int) = to, sorce(str) = tên folder chứa tập train, classify(arr) = mảng class muốn đọc

    #VD: getImage(1, 1000, "train", [1, 2, 3]) - Lấy 1000 ảnh từ tập "train" có nhãn là 1, 2 3

    def getImage(self, f = 0, t = None, sorce = "", classify=None):

        print("From: "+str(f))

        if t is None:

            print("To: All")

        else:

            print("To: ", str(t))

        if classify is not None:

            print("Classify = "+str(classify))

        print("Load Image "+sorce+"...")

        self.listFileFrom(sorce)

        arr = []

        label = []

        path = []

        if t is None:

            t = len(self.sourceFile)



        for i in range(f, t, 1):

            if classify is not None:

                for res in classify:

                    if self.sourceClassify[i] == res:

                        image = cv2.imread(self.sourceFile[i])

                        image = cv2.resize(image, (48, 48))

                        image = self.bgr(image)

                        arr.append(image)

                        path.append(str(self.sourceFile[i]))

                        label.append(self.unix[int(self.sourceClassify[i])])



            else:

                image = cv2.imread(self.sourceFile[i])

                path.append(str(self.sourceFile[i]))

                image = cv2.resize(image, (48, 48))

                image = self.bgr(image)

                arr.append(image)

                label.append(self.unix[int(self.sourceClassify[i])])





        return arr, label, path

    

    #Load ảnh từ tập public_test

    def getTest(self, f = 0, t = None, sorce = ""):

        print("From: " + str(f))

        if t is None:

            print("To: All")

        else:

            print("To: ", str(t))

        print("Load Image " + sorce + "...")

        self.listFileFrom(sorce)

        arr = []

        label = []

        if t is None:

            t = len(self.sourceFile)



        for i in range(f, t, 1):

            image = cv2.imread(self.sourceFile[i])

            image = cv2.resize(image, (48, 48))

            image = self.bgr(image)

            arr.append(image)

            label.append(os.path.basename(self.sourceFile[i]))

        return arr, label



    def test(self):

        label = []

        label.append(self.unix[1])

        print(label)





if __name__ == '__main__':

    t = ReadImage()

    t.test()
class Panel1(wx.Panel):

    def __init__(self, parent, id):

        wx.Panel.__init__(self, parent, id)

        try:

            bmp = wx.Image('../train2/0/1c682747f4bf4eeb8326d936eebacd6d.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()

            self.bitmap1 = wx.StaticBitmap(self, -1, bmp, (0, 0))

            parent.SetTitle("Year")

        except IOError:

            print("Not found")

            raise SystemExit



        self.button1 = wx.Button(self, id=-1, label='Change', pos=(50, 100))

        self.button1.Bind(wx.EVT_BUTTON, self.OnClicked)

        self.label_predict = wx.StaticText(self, label="-1", pos=(60, 0))

        self.target_label = wx.TextCtrl(self, pos=(60, 50))

        

    #Bắt sự kiện click

    def OnClicked(self, event):

        print("From: ", self.label_predict.GetLabel())

        val = int(self.target_label.GetValue())

        print("To:", t.pathLabel[val] + "\\f_" + os.path.basename(self.label_predict.GetLabel()))

        os.rename(self.label_predict.GetLabel(),

                  t.pathLabel[val] + "\\f_" + os.path.basename(self.label_predict.GetLabel()))

        

app = wx.App()

frame = wx.Frame(None, -1, "Image", size=(350, 400))



panel = Panel1(frame, -1)

frame.Show()



def setImage(link, text):

    print(link)

    bmp = wx.Image(link, wx.BITMAP_TYPE_ANY).ConvertToBitmap()

    panel.bitmap1 = wx.StaticBitmap(frame, -1, bmp, (0, 0))

    panel.label_predict = wx.StaticText(frame, label=text, pos=(60, 0))

    cv2.imshow("delay", 0)

    cv2.waitKey(0)

    

##Load model and check



app.MainLoop()
import tensorflow as tf

import main.ReadImg as ri

import numpy as np

import cv2



t = ri.ReadImage()

X_train = []

Y_train = []

X_test = []

Y_test = []

def shuffleData(X, Y):

    c = np.arange(X.shape[0])

    np.random.shuffle(c)

    return X[c], Y[c]



def loadData(X, Y, scaleTrain, scaleTest):

    total = scaleTrain+scaleTest

    p = round((len(X)/total)*scaleTrain)

    return (X[:p, :], Y[:p, :]), (X[p:, :], Y[p:, :])



unix = np.eye(43)



for i in range(43):

    #load image

    x, y, _ = t.getImage(sorce="train3", classify=[str(i)])

    x = np.array(x)

    y = np.array(y)

    #Trộn

    x, y = shuffleData(x, y)

    #Chia tập tỷ lệ 8-1

    (x_train, y_train), (x_test, y_test) = loadData(x, y, 8, 1)

    ax = []

    ay = []

    #Đa dạng ảnh

    for j in range(len(x_train)):

        ax.append(t.rotation(x_train[j], 18))

        ay.append(unix[i])

        ax.append(t.rotation(x_train[j], -18))

        ay.append(unix[i])

        

    x_train = np.append(x_train, np.array(ax), axis=0)

    y_train = np.append(y_train, ay, axis=0)

    

    if i == 0:

        X_train = x_train

        Y_train = y_train

        X_test = x_test

        Y_test = y_test

    else:

        X_train = np.append(X_train, x_train, axis=0)

        Y_train = np.append(Y_train, y_train, axis=0)

        X_test = np.append(X_test, x_test, axis=0)

        Y_test = np.append(Y_test, y_test, axis=0)



print("Data X = {}".format(X_train.shape))

print("Data Y = {}".format(Y_train.shape))



X_train = X_train.reshape(-1, 48, 48, 3)

X_test = X_test.reshape(-1, 48, 48, 3)

#Trộn tập train

X_train, Y_train = shuffleData(X_train, Y_train)
tf.reset_default_graph()

mX = tf.placeholder("float", [None, 48, 48, 3])

mY = tf.placeholder("float", [None, 43])

keep_prob = tf.placeholder(tf.float32)



def model(X, keep_prob):



    C1 = tf.layers.conv2d(X, 32, kernel_size=3, padding="VALID", activation=tf.nn.relu)

    P1 = tf.layers.max_pooling2d(C1, 2, 2, padding="VALID")

    D1 = tf.layers.dropout(P1, keep_prob)



    C2 = tf.layers.conv2d(D1, 64, kernel_size=4, padding="VALID", activation=tf.nn.relu)

    P2 = tf.layers.max_pooling2d(C2, 2, 2, padding="VALID")

    D2 = tf.layers.dropout(P2, keep_prob)



    C3 = tf.layers.conv2d(D2, 128, kernel_size=3, padding="VALID", activation=tf.nn.relu)

    P3 = tf.layers.max_pooling2d(C3, 2, 2, padding="VALID")

    D3 = tf.layers.dropout(P3, keep_prob)



    fc1 = tf.contrib.layers.flatten(D3)

    fc1 = tf.layers.dense(fc1, 200)



    output = tf.layers.dense(fc1, 43)



    return output



Y_pred = model(mX, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=mY))



optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

predict_op = tf.argmax(Y_pred, 1)
#Training....

epochs = 60



saver = tf.train.Saver()



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())



    for epoch in range(epochs):

        for start, end in zip(range(0, len(X_train), 128), range(128, len(X_train)+1, 128)):

            sess.run(optimizer, feed_dict={mX: X_train[start:end], mY: Y_train[start:end], keep_prob: 0.5})



        if epoch%10 == 0:

            arr = sess.run(predict_op, feed_dict={mX: X_test, keep_prob: 1.0})

            arr = np.array(arr)

            accuracy = np.mean(np.argmax(Y_test, axis=1) == arr)

            print("Epoch: {} and accuracy: {}".format(epoch, accuracy))

            save_path = saver.save(sess, "../tmp/model.ckpt")

            print("Model saved in path: %s" % save_path)

    print("Final: {}".format(np.mean(np.argmax(Y_test, axis=1) == sess.run(predict_op, feed_dict={mX: X_test, keep_prob:1.0}))))

    save_path = saver.save(sess, "../tmp/model.ckpt")

    print("Model saved in path: %s" % save_path)