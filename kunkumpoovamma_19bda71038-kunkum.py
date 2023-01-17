import pandas as pd  #handling the dataset
import numpy as np   #handling the numbers
import seaborn as sns
import matplotlib.pyplot as plt
mydata = pd.read_csv("C:/Users/Kunkum/Desktop/ML_EXAM/Train_Mask.csv")
mydata
mydata.isnull().sum().sum()
mydata.describe()
mydata.corr().abs()    #if dependant and independant is continuous
mydata.corr()['flag'].abs()
for i in mydata:
    mydata_plot = plt.plot(mydata[i])
mydata['flag'].value_counts()     #check class imbalance
#Graphical representation of class imbalance
sns.distplot(mydata['flag'],color='g');
#skewness and kurtosis
print("Skewness: %f" % mydata['flag'].skew())
print("Kurtosis: %f" % mydata['flag'].kurt())
from scipy import stats                                  #pointbiserial correlation when one variable is dichotomous
stats.pointbiserialr(mydata['flag'],mydata['timeindex'])
stats.pointbiserialr(mydata['flag'],mydata['currentBack'])
stats.pointbiserialr(mydata['flag'],mydata['motorTempBack'])
stats.pointbiserialr(mydata['flag'],mydata['positionBack'])
stats.pointbiserialr(mydata['flag'],mydata['refPositionBack'])
stats.pointbiserialr(mydata['flag'],mydata['refVelocityBack'])
stats.pointbiserialr(mydata['flag'],mydata['trackingDeviationBack'])
stats.pointbiserialr(mydata['flag'],mydata['velocityBack'])
stats.pointbiserialr(mydata['flag'],mydata['currentFront'])
stats.pointbiserialr(mydata['flag'],mydata['motorTempFront'])
stats.pointbiserialr(mydata['flag'],mydata['positionFront'])
stats.pointbiserialr(mydata['flag'],mydata['refPositionFront'])
stats.pointbiserialr(mydata['flag'],mydata['refVelocityFront'])
stats.pointbiserialr(mydata['flag'],mydata['trackingDeviationFront'])
stats.pointbiserialr(mydata['flag'],mydata['velocityFront'])
mydata.var(axis = 0)
#from scipy import stats
#import numpy as np
#z = np.abs(stats.zscore(mydata))
#print(z)
y = mydata[['flag']]
x = mydata.drop('flag',axis=1)
from sklearn.model_selection import train_test_split  #used to splitting training and test data
from sklearn.preprocessing import StandardScaler  #used for feature scaling
x_train,x_test,y_train,y_test = ms.train_test_split(x,y,test_size = 0.3,random_state = 0)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test =  sc_x.transform(x_test)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
fit = pca.fit(x)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)

from sklearn import svm

#Create a svm Classifier
classifier = svm.NuSVC(gamma="auto") # NONLinear Kernel 

#Train the model using the training sets
classifier.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = classifier.predict(x_test)
classifier.score(x_train, y_train)
classifier.score(x_test,y_test)
#y_pred = classifier.predict(x_test)
y_pred
from sklearn.metrics import classification_report, confusion_matrix 
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred))
from sklearn.metrics import f1_score
f1_score(y_test,y_pred)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred,y_test)
#confusionmatrix
from sklearn.metrics import confusion_matrix
confusion_matrix=pd.DataFrame(data=confusion_matrix(y_test,y_pred),columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize=(8,5))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap="winter_r")
data = pd.read_csv("C:/Users/Kunkum/Desktop/ML_EXAM/Test_Mask_Dataset.csv")
sc_x = StandardScaler()
data =  sc_x.fit_transform(data)
data.shape
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
fit = pca.fit(data)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)
y_preds =classifier.predict(data)
pr = pd.DataFrame(y_preds,columns=['flag'])
pr
ss = pd.read_csv("C:/Users/Kunkum/Desktop/ML_EXAM/SampleSubmission.csv")
ss.columns
ss['flag']=y_preds
ss.to_csv('C:/Users/Kunkum/Desktop/ML_EXAM/final3.csv',index = False)
