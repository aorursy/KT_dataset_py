import pandas as pd
df_train = pd.read_csv('iris_train.csv')
df_test = pd.read_csv('iris_test.csv')
df_train.head()
df_test.head()
df_train.columns
# features 是我們要分析的特徵
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
# target 是我們要分類的結果
target = ['Species']
x1 = df_train[features]
y1 = df_train[target]
x1.head()
x2 = df_test[features]
x2.head()
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x1.astype(float) # 將所有的欄位都轉成浮點數 casting to the float type
x2.astype(float)

x1 = ss.fit_transform(x1)
x2 = ss.transform(x2)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver='lbfgs')
classifier.fit(x1, y1)
y2 = classifier.predict(x2)
y2
#Create a DataFrame with the ids and our prediction regarding whether a flower is Iris-setosa or not
submission = pd.DataFrame({'id':df_test['id'],'Species':y2})

#Visualize the first 5 rows
submission.head()
#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'IRIS Predictions 2.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)