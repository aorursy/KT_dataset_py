# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option("display.max_columns", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 50)
train_data = pd.read_csv('../input/thai-mnist-classification/mnist.train.map.csv')

train_data['str_category']= train_data['category'].apply(lambda x: str(x))

Y = train_data[['category']]


kf = KFold(n_splits = 10)
                         
#skf = StratifiedKFold(n_split = 5, random_state = 7, shuffle = True) 
kf
idg = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.3,
                         fill_mode='nearest',
                         horizontal_flip = True,
                         rescale=1./255)
from tensorflow.keras import layers
import keras
import numpy as np
from keras.applications import InceptionResNetV2
from tensorflow.keras import layers
#Load the VGG19 model


ResNetV2 = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# fit input
x_ResNetV2 = ResNetV2.layers[-1].output
x_ResNetV2 = layers.GlobalAveragePooling2D()(x_ResNetV2)
x_ResNetV2 = layers.Dropout(0.5)(x_ResNetV2)
x_ResNetV2 = layers.Dense(10, activation='softmax')(x_ResNetV2)

model_ResNetV2 = keras.Model(ResNetV2.input, x_ResNetV2)
model_ResNetV2.summary()
VALIDATION_ACCURACY = []
VALIDATION_LOSS = []


image_dir='../input/thai-mnist-classification/train/'
fold_no = 1

num_epochs=10
batchsize= 32

callbacks_list =tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
model_ResNetV2.compile(loss='CategoricalCrossentropy',optimizer= keras.optimizers.Adam(learning_rate=0.00005),metrics=['CategoricalAccuracy'])



for train_index, val_index in kf.split(np.zeros(len(train_data)),Y):
    training_data = train_data.iloc[train_index]
    validation_data = train_data.iloc[val_index]
    
    train_data_generator = idg.flow_from_dataframe(training_data, directory = image_dir,
        x_col = "id", y_col = "str_category",
        class_mode = "categorical", shuffle = True)
    valid_data_generator  = idg.flow_from_dataframe(validation_data, directory = image_dir,
        x_col = "id", y_col = "str_category",
        class_mode = "categorical", shuffle = True)


# FIT THE MODEL
    history = model_ResNetV2.fit(train_data_generator,epochs=num_epochs,batch_size= batchsize,callbacks=callbacks_list,validation_data=valid_data_generator,verbose=1)
    results = model_ResNetV2.evaluate(valid_data_generator)
    
    print(f'Score for fold {fold_no}: {model_ResNetV2.metrics_names[0]} of {results[0]}; {model_ResNetV2.metrics_names[1]} of {results[1]}%')

    VALIDATION_ACCURACY.append(results[0])
    VALIDATION_LOSS.append(results[1])

    fold_no += 1
import os
folder_path='../input/thai-mnist-classification/test/'
images = []
yhat=[]
for i in os.listdir(folder_path):
    img = os.path.join(folder_path, i)
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)/255.
    img = np.expand_dims(img, axis=0)
    y=model_ResNetV2.predict(img).argmax(axis=-1)
    yhat.append([i,y])
yhat
df_yhat=pd.DataFrame(yhat)
df_yhat.rename(columns={0:'id',1:'category'},inplace=True)
df_yhat.head(5)
df_yhat['category']=df_yhat['category'].apply(lambda x : x[0])
train_rule =pd.read_csv('../input/thai-mnist-classification/train.rules.csv')
train_rule.rename(columns={'id':'ref_id'},inplace=True)
train_rule.head(3)
train_data.head()
train_rule_rev =pd.merge(train_rule,train_data, left_on='feature1',right_on='id' ,how='left')
train_rule_rev1 =pd.merge(train_rule_rev,train_data, left_on='feature2',right_on='id' ,how='left')
train_rule_rev2 =pd.merge(train_rule_rev1,train_data, left_on='feature3',right_on='id' ,how='left')
train_rule_final = train_rule_rev2[['ref_id','predict','category_x','category_y','category']].rename(columns={'category_x':'new_feature1','category_y':'new_feature2','category':'new_feature3'})
train_rule_final.head()
train_rule_final['new_feature1']=train_rule_final['new_feature1'].fillna(10)
train_rule_final['new_feature1']=train_rule_final['new_feature1'].astype('category')
def one_hot_encoder(dataframe):

  # Select category columns
  cat_df = dataframe.select_dtypes(include=['category']).columns.to_list()

  # Convert to one-hot dataframe
  one_hot_df = pd.get_dummies(dataframe, columns=cat_df, drop_first=True)
  
  return one_hot_df
train_rule_final = one_hot_encoder(train_rule_final)
train_rule_final
test_rule =pd.read_csv('../input/thai-mnist-classification/test.rules.csv')
test_rule.rename(columns={'id':'ref_id'},inplace=True)
test_rule.head(3)
df_yhat.head(10)
test_rule_rev =pd.merge(test_rule,df_yhat, left_on='feature1',right_on='id' ,how='left')
test_rule_rev1 =pd.merge(test_rule_rev,df_yhat, left_on='feature2',right_on='id' ,how='left')
test_rule_rev2 =pd.merge(test_rule_rev1,df_yhat, left_on='feature3',right_on='id' ,how='left')
test_rule_final = test_rule_rev2[['ref_id','category_x','category_y','category']].rename(columns={'category_x':'new_feature1','category_y':'new_feature2','category':'new_feature3'})
test_rule_final['new_feature1']=test_rule_final['new_feature1'].fillna(10)
test_rule_final['new_feature1']=test_rule_final['new_feature1'].astype('category')
test_rule_final_onehot = one_hot_encoder(test_rule_final)
test_rule_final_onehot.head(3)
test_rule_final_onehot=test_rule_final_onehot.drop(columns='ref_id')
test_rule_final_onehot
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,confusion_matrix
train_rule_final
X=train_rule_final.drop(columns=['ref_id','predict'])
y=train_rule_final['predict']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30)
reg = LinearRegression()
reg.fit(X_train,y_train)

xgb = XGBRegressor()
xgb.fit(X_train,y_train)

greg = GradientBoostingRegressor(random_state=0)
greg.fit(X_train,y_train)

rm = RandomForestRegressor(max_depth=6)
rm.fit(X_train,y_train)
pred_1 = reg.predict(X_test)
pred_2 = xgb.predict(X_test)
pred_3  = greg.predict(X_test)
pred_4  = rm.predict(X_test)
#Create Function Evaluate Model
import numpy as np
from statsmodels.tsa.stattools import acf
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    #acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse})
            #'corr':corr, 'minmax':minmax})

#หาค่า Accuracy ของ Model Arima Long Period
print("linear regression Metric")
print(forecast_accuracy(y_test, pred_1))
print('\n')
print("Xgboost regression Metric")
print(forecast_accuracy(y_test, pred_2))
print('\n')
print("Gradient Bossting regression Metric")
print(forecast_accuracy(y_test, pred_3))
print('\n')
print("Random Forest regression Metric")
print(forecast_accuracy(y_test, pred_4))
test_rule_predict=xgb.predict(test_rule_final_onehot)
test_rule_predict_df=pd.DataFrame(test_rule_predict)
test_rule_predict_df=test_rule_predict_df.rename(columns={0:'predict'})
from decimal import Decimal, ROUND_HALF_UP
test_rule_predict_df['predict']=test_rule_predict_df['predict'].apply(lambda x: Decimal(x).to_integral_value(rounding=ROUND_HALF_UP))
test_rule_predict_df.groupby('predict').size()
test_rule
test_rule_predict_result=pd.concat([test_rule_final,test_rule_predict_df],axis=1)
test_rule_predict_result=test_rule_predict_result.rename(columns={'ref_id':'id'})
test_rule_predict_result['predict']=test_rule_predict_result['predict'].apply(lambda x : 0 if x <0 else x)
test_rule_predict_result.groupby('predict').size()
test_rule_predict_result.to_csv('22p22w0030_W4Q1_1.csv',index=False)
