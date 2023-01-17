
import numpy as np # linear algebra
import pandas as pandas # data processing, CSV file I/O (e.g. pd.read_csv)
#########
import seaborn as sns

import matplotlib

import numpy as numpy
import pandas as pandas
import statsmodels.api
import statsmodels.formula.api as smf
import statsmodels.api as sm

import statsmodels.stats.multicomp as multi 

import scipy
import matplotlib.pyplot as plt
import warnings 

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.ensemble import ExtraTreesClassifier

warnings.simplefilter(action = "ignore", category = FutureWarning) 
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#Reading the data
data1 = pandas.read_csv("../input/Speed Dating Data.csv", encoding="ISO-8859-1")

##Selecting Only the Relevant Variables for the Analysis
temp1=data1[['iid','gender','pid','samerace','age_o','race_o','dec_o','attr_o','sinc_o','intel_o','fun_o','amb_o','shar_o','like_o','prob_o','age','field_cd','race','imprace','imprelig','from','date','go_out','dec','attr','sinc','intel','fun','amb','shar','like','prob']]
###################################################################################################
# The next lines are to have on the same raw all the relevant information from both the partners.##
####################################################################################################

#Creation DataSet for Merging Missing Variables
temp2=temp1[['iid','field_cd','imprace','imprelig','from','date','go_out']]
#Rename the variables to avoid confusion with the two data frames...
temp2.columns = ['pid','field_cd_o','imprace_o','imprelig_o','from_0','date_0','go_out_o']
#Merge the two datasets to have all the variables for both the partners.
BothGenders=pandas.merge(temp1,temp2,on='pid')

BothGenders=BothGenders.drop('iid',1)
BothGenders=BothGenders.drop('pid',1)
BothGenders=BothGenders.dropna()
###############################################################
#Creation New Features to further analysis potential patterns##
###############################################################
#Difference of the age between the parther instead of the "absolute" age.  
BothGenders['Delta_Age']=BothGenders['age'] - BothGenders['age_o']
#Same field of career
BothGenders['SameField']=BothGenders['field_cd'] == BothGenders['field_cd_o']
#Provenience from the state.
BothGenders['SameState']=BothGenders['from'] == BothGenders['from_0']
BothGenders=BothGenders.drop('from',1)
BothGenders=BothGenders.drop('from_0',1)
#Subset the dataframe for the two genders From now on we will use only these two datasets.
Females=BothGenders.loc[BothGenders['gender'] == 0]
Males=BothGenders.loc[BothGenders['gender'] == 1]
#Average for all the Features Group by 'dec' factor
#Females
Females[Females.columns[:]].groupby(Females['dec']).mean().round(2)
#Males
Males[Males.columns[:]].groupby(Males['dec']).mean().round(2)

model1 = smf.ols(formula='dec ~ C(samerace)+age_o+C(race_o)+dec_o+attr_o+sinc_o+intel_o+fun_o+amb_o+shar_o+like_o+prob_o+imprace+imprelig+date+go_out+attr+sinc+intel+fun+amb+shar+like+prob+age+age_o+Delta_Age+go_out_o+date_0+C(race)', data=Females)
results1 = model1.fit()

table = sm.stats.anova_lm(results1, typ=2) 

FeaturesImportance=sorted(zip(table.F,table.index),reverse=True)
dfFemales = pandas.DataFrame(FeaturesImportance, columns=['Model.Feature_Importances_Based_on_F', 'predictors.columns'])
print("Top 10 Features with the highest F value") 
print(dfFemales.head(10).round(2))
ax0=sns.barplot(y="predictors.columns", x="Model.Feature_Importances_Based_on_F", data=dfFemales,
             palette="Blues")
ax0.set(ylabel='Predictors', xlabel='F value',title="Female Group, F values for each Predictor")
plt.show()
Females.boxplot(column=['like','attr','attr_o','shar','prob'], by=['dec'])
print("if the partner are from the same race are more keen to go for a date?")
pandas.crosstab(Females.samerace,Females.dec).apply(lambda r: r/r.sum(), axis=1).round(2)
print("what are the cross selections from the different races ")
pandas.crosstab([Females.race,Females.race_o],Females.dec).apply(lambda r: r/r.sum(), axis=1).round(2)
model1 = smf.ols(formula='dec ~ C(samerace)+age_o+C(race_o)+dec_o+attr_o+sinc_o+intel_o+fun_o+amb_o+shar_o+like_o+prob_o+imprace+imprelig+date+go_out+attr+sinc+intel+fun+amb+shar+like+prob+age+age_o+Delta_Age+go_out_o+date_0+C(race)', data=Males)
results1 = model1.fit()

table = sm.stats.anova_lm(results1, typ=2)
FeaturesImportance=sorted(zip(table.F,table.index),reverse=True)
dfMales = pandas.DataFrame(FeaturesImportance, columns=['Model.Feature_Importances_Based_on_F', 'predictors.columns'])
print("Top 10 Features with the highest F value") 
print(dfMales.head(10))
ax1=sns.barplot(y="predictors.columns", x="Model.Feature_Importances_Based_on_F", data=dfMales,palette="Blues")
ax1.set(ylabel='Predictors', xlabel='F value',title="Male Group, F values for each Predictor")
plt.show()
Males.boxplot(column=['like','attr','attr_o','shar','prob'], by=['dec'])
pandas.crosstab(Males.samerace,Males.dec).apply(lambda r: r/r.sum(), axis=1).round(2)
pandas.crosstab([Males.race,Males.race_o],Males.dec ).apply(lambda r: r/r.sum(), axis=1).round(2)
predictorsF = Females.drop('dec',1)
predictorsF = predictorsF.drop('gender',1)
targetsF = Females.dec
pred_trainF, pred_testF, tar_trainF, tar_testF  =   train_test_split(predictorsF, targetsF, test_size=.4)
    
classifierF=DecisionTreeClassifier(max_depth=4)
classifierF=classifierF.fit(pred_trainF,tar_trainF)
   
predictionsF=classifierF.predict(pred_testF)
print
print("Confusion Matrix")
print(sklearn.metrics.confusion_matrix(predictionsF,tar_testF))
print("Accuracy Score")
print(sklearn.metrics.accuracy_score(predictionsF,tar_testF))
from sklearn import tree

######################
from sklearn.externals.six import StringIO
with open("Female.dot", 'w') as f:
     f = tree.export_graphviz(classifierF, out_file=f,
                             feature_names=predictorsF.columns,
                             filled=True, rounded=True,special_characters=True)
        #     dot -Tpng "C:\Users\enzo7311\Dropbox\Public\dataAnalysis\capstone\Female.dot" > c:\temp\Female.png
        
      #As Soon As i understand how to visualize image saved externally I will include the Class Tree picture  
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
 
fpr, tpr, _ = metrics.roc_curve(tar_testF, predictionsF)

roc_auc = auc(fpr, tpr)
print ('ROC AUC: %0.2f' % roc_auc)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
#from IPython.display import Image
#Image(filename='c:/temp/female.png')
predictorsM = Males.drop('dec',1)
predictorsM = predictorsM.drop('gender',1)

targetsM = Males.dec
pred_trainM, pred_testM, tar_trainM, tar_testM  =   train_test_split(predictorsM, targetsM, test_size=.4)
    
classifierM=DecisionTreeClassifier(max_depth=4)
classifierM=classifierM.fit(pred_trainM,tar_trainM)
   
predictionsM=classifierM.predict(pred_testM)
print
   
print("Confusion Matrix")
print(sklearn.metrics.confusion_matrix(predictionsM,tar_testM))
print("Accuracy Score")
print(sklearn.metrics.accuracy_score(predictionsM,tar_testM))

with open("Males.dot", 'w') as f:
     f = tree.export_graphviz(classifierM, out_file=f,
                             feature_names=predictorsM.columns,
                             filled=True, rounded=True,special_characters=True)
                             #class_names=targets.columns)
##
#dot -Tpng "C:\Users\enzo7311\Dropbox\Public\dataAnalysis\capstone\Males.dot" > c:\temp\Males.png
##     
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
 
fpr, tpr, _ = metrics.roc_curve(tar_testM, predictionsM)

roc_auc = auc(fpr, tpr)
print ('ROC AUC: %0.2f' % roc_auc)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
#Image(filename='c:/temp/males.png')
from sklearn.ensemble import RandomForestClassifier

classifierF=RandomForestClassifier(n_estimators=25)
classifierF=classifierF.fit(pred_trainF,tar_trainF)

predictionsF=classifierF.predict(pred_testF)

sklearn.metrics.confusion_matrix(tar_testF,predictionsF)
sklearn.metrics.accuracy_score(tar_testF, predictionsF)


# fit an Extra Trees model to the data
modelF = ExtraTreesClassifier()
modelF.fit(pred_trainF,tar_trainF)
print("Confusion Matrix")
print(sklearn.metrics.confusion_matrix(predictionsF,tar_testF))
print("Accuracy Score")
print(sklearn.metrics.accuracy_score(predictionsF,tar_testF))

FeaturesImportanceF=sorted(zip( modelF.feature_importances_,predictorsF.columns))
dfFemales = pandas.DataFrame(FeaturesImportanceF, columns=['model.feature_importances_', 'predictors.columns'])

ax3=sns.barplot(y="predictors.columns", x="model.feature_importances_", data=dfFemales,
             palette="Blues")
ax3.set(ylabel='Predictors', xlabel='Importance Attribute',title="Female Group, Importance for each Predictor")
plt.show(ax3)

classifierM=RandomForestClassifier(n_estimators=25)
classifierM=classifierM.fit(pred_trainM,tar_trainM)

predictionsM=classifierM.predict(pred_testM)

sklearn.metrics.confusion_matrix(tar_testM,predictionsM)
sklearn.metrics.accuracy_score(tar_testM, predictionsM)


# fit an Extra Trees model to the data
modelM = ExtraTreesClassifier()
modelM.fit(pred_trainM,tar_trainM)
print("Confusion Matrix")
print(sklearn.metrics.confusion_matrix(predictionsM,tar_testM))
print("Accuracy Score")
print(sklearn.metrics.accuracy_score(predictionsM,tar_testM))

FeaturesImportanceM=sorted(zip( modelM.feature_importances_,predictorsM.columns))
dfMales = pandas.DataFrame(FeaturesImportanceM, columns=['model.feature_importances_', 'predictors.columns'])

ax4=sns.barplot(y="predictors.columns", x="model.feature_importances_", data=dfMales, palette="Blues")
ax4.set(ylabel='Predictors', xlabel='Importance Attribute',title="Males Group, Importance for each Predictor")
plt.show(ax4)