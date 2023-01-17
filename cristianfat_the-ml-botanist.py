import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
import os
plt.style.use('seaborn')
p = '/kaggle/input/plant-seedlings-classification'

def df_of_images(folder_name,path='/kaggle/input/plant-seedlings-classification'):
    itms = list()
    for x in os.listdir(os.path.join(path,folder_name)):
        for img in os.listdir(os.path.join(path,folder_name,x)):
            itms.append({
                'label': x.lower().strip().replace(' ','_').replace('-','_'),
                'image_path': os.path.join(path,folder_name,x,img)
            })
    return pd.DataFrame(itms)
        
train = df_of_images('train')
test = pd.DataFrame({ 'image_path': [ os.path.join(p,'test',i) for i in os.listdir('/kaggle/input/plant-seedlings-classification/test')]})
train.label.value_counts().plot.bar(rot=0,figsize=(25,7));
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
base_model = InceptionV3()
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)	
def extract_features_keras(image_path,model):
	img = image.load_img(image_path, target_size=(299, 299))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	predictions = model.predict(x)
	return np.squeeze(predictions)
%%time
train['image_features'] = train.image_path.apply(lambda x: extract_features_keras(x,base_model) )
test['image_features'] = test.image_path.apply(lambda x: extract_features_keras(x,base_model) )
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
train_,test_ = train_test_split(train,test_size=0.33,random_state=42,stratify=train.label)
'train:',train_.label.value_counts() / len(train_),'test:',test_.label.value_counts() / len(test_)
%%time
xgc = xgb.XGBClassifier(objective='multi:softmax',num_class=train.label.nunique())
xgc.fit(pd.DataFrame(train_['image_features'].values.tolist()),train_.label)
results = test_.copy()
results['y_pred'] = xgc.predict(pd.DataFrame(test_['image_features'].values.tolist()))
print(metrics.classification_report(results.label,results.y_pred))
sns.heatmap(metrics.confusion_matrix(results.label,results.y_pred),annot=True,fmt='d');
%%time
xgc = xgb.XGBClassifier(objective='multi:softmax',num_class=train.label.nunique())
xgc.fit(pd.DataFrame(train_['image_features'].values.tolist()),train_.label)
label_map = {x.lower().strip().replace(' ','_').replace('-','_'):x for x in os.listdir(os.path.join(p,'train'))}
label_map
%%time
results = test.copy()
results['species'] = xgc.predict(pd.DataFrame(test['image_features'].values.tolist()))
results['species'] = results['species'].replace(label_map)
results['file'] = results.image_path.apply(lambda x: x.split('/')[-1])
results[['file','species']]
results[['file','species']].to_csv('submission.csv',index=False)