import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from scipy import stats
data = pd.read_csv('C:\\Users\\id\Desktop\\Train_Mask.csv')#train data set
data_test = pd.read_csv('C:\\Users\\id\\Desktop\\Test_Mask_Dataset.csv')# test data set
data_miss_val = data.isnull().sum()#to get the null values
                                   #found that there are no null values.
summary_stat = data.describe().transpose()#summary statistics
a=data.groupby('flag').flag.count()/len(data['flag'])
sns.distplot(data['flag']);
#skewness and kurtosis
print("Skewness: %f" % data['flag'].skew())
print("Kurtosis: %f" % data['flag'].kurt())  #there is class imbalance

#correlation
corr=data.corr()
corr
plt.subplots(figsize=(10,4))
sns.heatmap(corr, annot=True)
#handling outliers
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(data))
#print(z)
threshold = 3
print(np.where(z > 3))
data = data[(z < 3).all(axis=1)]

Y=data['flag']#predicting variables
X = data.drop('flag',axis=1)#response variables
data_1= data_test
import sklearn.model_selection as ms
import sklearn.preprocessing as pre
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler
#splitting the train set for  validation.
X_train,X_test,Y_train,Y_test = ms.train_test_split(X,Y,test_size=0.2,random_state=0)
X_train.shape,X_test.shape,Y_test.shape,Y_train.shape
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()# to normalise the data
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)
data_1 = scale.fit_transform(data_1)
from sklearn.decomposition import PCA#to reduce the dimension PCA is used
pca = PCA(n_components=1)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)
from sklearn import svm

#Create a svm Classifier
classifier = svm.NuSVC(gamma='auto') # NONLinear Kernel 

#Train the model using the training sets
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
classifier.score(X_train,Y_train)
classifier.score(X_test,Y_test)
from sklearn.metrics import f1_score
F1_score = f1_score(Y_test,Y_pred)
F1_score
#confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize=(8,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")

#prediction
predict = classifier.predict(data_1)
df = pd.DataFrame(Y_pred,columns=['flag'])

sample = pd.read_csv('C:\\Users\\id\\Desktop\\Sample Submission.csv')
sample['flag'] = predict
sample.to_csv('C:\\Users\\id\\Desktop\\submit_6.csv',index = False)

#import sklearn.linear_model as lm
#log_reg=lm.LogisticRegression(solver='lbfgs',max_iter=1000)
#log_reg.fit(X_train,Y_train)
#log_reg.score(X_train,Y_train)#Accuracy
#log_reg.score(X_test,Y_test)

