import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.utils import shuffle
import keras

df_csv = pd.read_csv("../input/creditcard.csv")
df_csv.head()
df_norm = (df_csv - df_csv.min() ) / (df_csv.max() - df_csv.min() )
df_norm.head()
# 전체거래건/사기건 비율
ratio = len(df_norm)/len(df_norm[df_norm.Class!=0.0]) 
df_norm.loc[:,'Class'] *= ratio
df_norm[df_norm.Class!=0.0][:3]
# 학습 80% 평가 20%
df_norm_train,df_norm_test = np.split(df_norm,[int(0.8*len(df_norm))])  

x_train = df_norm_train.loc[:,'V1':'Amount']
y_train = df_norm_train.loc[:,'Class']

x_test = df_norm_test.loc[:,'V1':'Amount']
y_test = df_norm_test.loc[:,'Class']
#모델 구성하기
model = Sequential()
model.add(Dense(32,input_dim=len(x_train.columns),activation="relu"))
model.add(Dropout(0.3))
for i in range(2):
    model.add(Dense(32,activation="relu"))
    model.add(Dropout(0.3))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#모델 컴파일하기
model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'])
#텐서보드 설정
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
#모델 학습하기
hist = model.fit(x_train, y_train, epochs=5, batch_size=64, callbacks=[tb_hist])
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
# # 모델 평가하기
# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
# print('')
# print('loss_and_metrics : ' + str(loss_and_metrics))
# 모델 평가하기
scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" %(model.metrics_names[0], scores[1]*100))
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
#true positives, false negative 구하기

#true만 추출
df_positive = df_norm_test[df_norm_test.Class!=0.0]

# round predictions
prediction_positive = model.predict(df_positive.loc[:,'V1':'Amount'])
rounded_positive = [round(x[0]) for x in prediction_positive]

#true positives
tp = rounded_positive.count(1.0)
#false negative
fn = rounded_positive.count(0.0)
print("tp : ",tp)
print("fn : ",fn)
print("positive : ", len(df_positive))
#false positive, true negative 구하기

#false만 추출
df_negative = df_norm_test[df_norm_test.Class==0.0]

# round predictions
prediction_negative = model.predict(df_negative.loc[:,'V1':'Amount'])
rounded_negative = [round(x[0]) for x in prediction_negative]

#false positive
fp = rounded_negative.count(1.0)
#true negative
tn = rounded_negative.count(0.0)
print("fp : ",fp)
print("tn : ",tn)
print("negative : ", len(df_negative))
# http://bcho.tistory.com/1206 참고
# recall(민감도) 측정
recall = tp/(tp+fn)
print("recall = tp/(tp+fn) : ", recall)

#precision(정밀성) 측정
precision = tp/(tp+fp)
print("precision = tp/(tp+fp) : ", precision)

#accuracy(정확도) 측정
accuracy = (tp+tn)/(len(df_negative) + len(df_positive))
print("accuracy = (tp+tn)/(len(df_negative) + len(df_positive)) : ", accuracy)

#f1 score
f1_score = (2*precision*recall)/(precision+recall)
print("f1 score = (2*precision*recall)/(precision+recall) : ", f1_score)
