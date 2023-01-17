import pandas as pd

import numpy as np

from sklearn.metrics import confusion_matrix

from sklearn.metrics import auc,roc_curve
laCare_df=pd.read_csv("../input/model_outcome.csv.txt")
laCare_df
predicted_prob=laCare_df['predicted_prob']
### Converting predicted values into classes using threshold

threshold=0.5

predicted_prob_class1=np.zeros(predicted_prob.shape)

predicted_prob_class1[predicted_prob>threshold]=1

cm1 = confusion_matrix(laCare_df[['class']],predicted_prob_class1)

total=sum(sum(cm1))



#####from confusion matrix calculate accuracy

accuracy=(cm1[0,0]+cm1[1,1])/total

print("Manually calculated accuracy at threshold 0.5: ",accuracy)



#####Calculated sensitivity

sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])

print("Manually calculated sensitivity at threshold 0.5: ",sensitivity)





#####Calculated specificity

specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

print("Manually calculated specificity at threshold 0.5: ",specificity)
#the one calculated from inbuilt function

#from sklearn.metrics import roc_curve

#auc(fpr,tpr)
######manually calculated AUC



my_ts=np.linspace(0,1,100)

myres_fpr=[]

myres_tpr=[]

myres_thres=[]

for i in my_ts:

    myres_thres.append(i)

    predicted_prob_class1=np.zeros(predicted_prob.shape)

    predicted_prob_class1[predicted_prob>i]=1

    cm = confusion_matrix(laCare_df[['class']],predicted_prob_class1)

    total1=sum(sum(cm))

    #####from confusion matrix calculate accuracy

    accuracy1=(cm[0,0]+cm[1,1])/total1

    sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])

    myres_tpr.append(sensitivity1)

    specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])

    fpr=1-specificity1

    myres_fpr.append(fpr)

   

print("Manually calculated AUC:",auc(myres_fpr,myres_tpr))    
#plotted ROC



import sklearn.metrics as metrics

fpr,tpr,threshold=roc_curve(laCare_df['class'],laCare_df['predicted_prob'], pos_label=1)

roc_auc = metrics.auc(fpr, tpr)

from matplotlib import pyplot

# plot abline

pyplot.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

pyplot.plot(fpr, tpr, marker='.')

# show the plot

pyplot.show()