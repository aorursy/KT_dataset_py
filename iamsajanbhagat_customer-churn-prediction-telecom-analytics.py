import pandas as pd
import numpy as np
import warnings
import ipywidgets as widgets
from IPython.display import display
import plotly.express as px
import plotly.offline as py#visualization
import cufflinks
from scipy.stats import chi2_contingency

py.init_notebook_mode(connected=True)#visualization


warnings.filterwarnings(action="ignore")
Data_Path="https://raw.githubusercontent.com/bhagatsajan0073/Community-of-Practice/master/Hands%20on%20Session/Telecom_Churn_Dataset.csv"
telecomData=pd.read_csv(Data_Path)
telecomData.head()
print("No. of Records Present in Data : ",telecomData.shape[0])
print("No. of Attributes Present in Data : ",telecomData.shape[1])
telecomData.columns
"""   Unique Identifier in Data  """ 
uniqueIdentifier="phone number"


"""   Target Variable """
targetVariable="churn"


"""   Categorical Varibles /Qualitative variables"""
categoricalVariables=[
    "state","area code","international plan",'voice mail plan'
]

"""   Numerical Variables / Quantative Variables"""
numericalVariables =[
    i for i in telecomData.columns if((i not in [uniqueIdentifier,targetVariable]) & (i not in categoricalVariables))
]
print("Numerical Variables : {0}".format(", ".join(numericalVariables)))
churn_stats=(telecomData[targetVariable].value_counts()*100/telecomData.shape[0]).reset_index()
churn_stats.columns=['Churn Flag','Churn Percentage']

churn_stats

px.pie(churn_stats,
  names="Churn Flag", 
  values="Churn Percentage",title="Customer Attrition Rate"
)
catgorialDropDown=widgets.Dropdown(
    options=categoricalVariables,
    value=categoricalVariables[0],
    description='Select Attribute :',
    disabled=False,
)

categoricalOutput=widgets.Output()

for col in categoricalVariables:
    telecomData[col]=telecomData[col].apply(str)

def summary_dropdown_eventhandler(change):
    
    """Event Handler for Cateorical Variable Distribution Analysis"""
    
    categoricalOutput.clear_output()
    
    with categoricalOutput:
        temp_df=telecomData.pivot_table(index=change.new,
                                columns=[targetVariable],
                                values=uniqueIdentifier,aggfunc=len)
        temp_df.columns=['False','True']
        temp_df['sum']=temp_df.sum(axis=1)
        temp_df['churn_percentage']=temp_df['True']*100/temp_df['sum']
        temp_df.sort_values("churn_percentage",ascending=False,inplace=True)
        display(temp_df[['churn_percentage']].iplot(asFigure=True,
            kind="bar",
            xTitle=change.new,
            yTitle="Churn Percentage",
            title="Churn Percentage by {0}".format(change.new.capitalize())
        ))
        display(temp_df[temp_df['churn_percentage']>15].index)

        
catgorialDropDown.observe(summary_dropdown_eventhandler,names="value")

display(catgorialDropDown)
display(categoricalOutput)
df=telecomData.pivot_table(index='area code',
                   columns=targetVariable,
                   values=uniqueIdentifier,aggfunc=len)
df.columns=['False','True']
chi, pval, dof, exp = chi2_contingency(df)

print("Orignal Vales (O) : ")
df
print("Columns Total :")
df.sum(axis=0) ## columns sum
print("Row Total")
df.sum(axis=1) ## row sum
print("Grand Total")
df.sum().sum() ### Grand Total
contigency_table=[
    [(838*2850)/3333,(838*483)/3333],
    [(1655*2850)/3333,(1655*483)/3333],
    [(840*2850)/3333,(840*483)/3333],
]

print(contigency_table)

assert((contigency_table==exp).sum())
chi_value_calculated=np.divide(np.power(df.to_numpy()-contigency_table,2),exp).sum()
chi_value_calculated,pval
assert(chi_value_calculated==chi)
nrow=df.shape[0]
ncol=df.shape[1]
print("Degree of freedom : ",(nrow-1)*(ncol-1))
catgorialDropDown=widgets.Dropdown(
    options=categoricalVariables,
    value=categoricalVariables[0],
    description='Select Attribute :',
    disabled=False,
)

# categoricalOutput=widgets.Output()

def chi2_dropdown_eventhandler(selection):
    
    """Event Handler for Cateorical Variable Distribution Analysis"""
    
#     categoricalOutput.clear_output()
    
#     with categoricalOutput:
    df=telecomData.pivot_table(index=selection,
                       columns=targetVariable,
                       values=uniqueIdentifier,aggfunc=len)
    df.columns=['False','True']
    chi, pval, dof, exp = chi2_contingency(df)
    significance = 0.05
    
    print('p-value=%.6f, significance=%.2f\n' % (pval, significance))
    if pval < significance:
        print("""At %.2f level of significance, we reject the null hypotheses and accept H1.\n%s and %s are not independent.""" % (significance,targetVariable,selection))
    else:
        print("""At %.2f level of significance, we accept the null hypotheses.\n%s and %s are independent.""" % (significance,targetVariable,selection)
               )
    print()
    
    
# catgorialDropDown.observe(chi2_dropdown_eventhandler,names="value")
out = widgets.interactive_output(chi2_dropdown_eventhandler, {'selection': catgorialDropDown})

display(catgorialDropDown)
display(out)


numericalDropDown=widgets.Dropdown(
    options=numericalVariables,
    value=numericalVariables[0],
    description='Select Attribute :',
    disabled=False,
)

numericalOutput=widgets.Output()

def numerical_dropdown_eventhandler(change):
    """Event Handler for Numerical Variables Distribution Analysis"""
    numericalOutput.clear_output()
    
    with numericalOutput:
        
        figHistSingle=px.histogram(telecomData[telecomData[change.new]>0],
                   x=change.new)
        
        figHist = px.histogram(telecomData[telecomData[change.new]>0],
                   x=change.new, color=targetVariable)
       
        figBoxSingle=px.box(telecomData[telecomData[change.new]>0], 
                        y=change.new,
                       )
        
        figBox = px.box(telecomData[telecomData[change.new]>0], 
                        x=targetVariable, y=change.new,
                       )
        
        
        figHistSingle.update_layout(height=400, width=600).show()
        figHist.update_layout(height=400, width=600).show()
        figBoxSingle.update_layout(height=400, width=600).show()
        figBox.update_layout(height=400, width=600).show()
        
        
numericalDropDown.observe(numerical_dropdown_eventhandler,names="value")

display(numericalDropDown)
display(numericalOutput)
# figBox = px.box(telecomData, 
#                 y="total night calls")
# figBox.show()
telecomData[(telecomData['total night calls']>=155)|
            (telecomData['total night calls']<=46)].shape[0]
# Custom Correlation Function
def correlation(x,y):
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    return np.round(np.sum(np.multiply((x-x_mean),
                              (y-y_mean)))/np.sqrt(np.sum((x-x_mean)**2
                                                         )*np.sum((y-y_mean)**2)),3)
correlation(telecomData['customer service calls'],telecomData[targetVariable])
import random
randomlist = random.sample(range(0, telecomData.shape[0]), telecomData.shape[0])
randomlist=np.array(randomlist)/10

var1=telecomData['total day charge']
var2=telecomData['total day charge']*-np.random.randint(10)+randomlist

print("Correlation : ",correlation(var1,var2))

df=pd.DataFrame({
    "Var 1":var1,
    "Var 2":var2
})

df.iplot(asFigure=True, kind ='scatter', x ='Var 1', y ='Var 2', mode ='markers')
var1=telecomData['total day charge']
var2=telecomData['total day charge']*np.random.randint(10)+randomlist

print("Correlation : ",correlation(var1,var2))

df=pd.DataFrame({
    "Var 1":var1,
    "Var 2":var2
})

df.iplot(asFigure=True, kind ='scatter', x ='Var 1', y ='Var 2', mode ='markers')
added_cols=numericalVariables.copy()
added_cols.append(targetVariable)
correlationData=telecomData[added_cols].corr()
import plotly.figure_factory as ff

x = list(correlationData.columns)
y = list(correlationData.index)

fig = ff.create_annotated_heatmap(
    np.round(correlationData.to_numpy(),2), x=x, y=y
)

fig.show()
DropDown1=widgets.Dropdown(
    options=added_cols,
    value="total day charge",
    description='Select Tenure :',
    disabled=False,
)

DropDown2=widgets.Dropdown(
    options=added_cols,
    value="total day minutes",
    description='Select Question :',
    disabled=False,
)

ui = widgets.VBox([DropDown1, DropDown2])

def getCorrelation(var1,var2):
    df=telecomData[[var1,var2]]
    display(df.iplot(asFigure=True,
                     kind ='scatter',
                     x =var1,
                     y =var2, 
                     mode ='markers',
                     xTitle=var1,
                     yTitle=var2,
                     title="Scatter Plot for {0} vs {1}".format(var1,var2)
                    ))

out = widgets.interactive_output(getCorrelation, {'var1': DropDown1,
                                                  'var2': DropDown2})

display(ui,out)
# !pip install ExploriPy
# from ExploriPy import EDA
# CategoricalFeatures = ['state','area code','phone number','international plan','voice mail plan','churn']
# eda = EDA(telecomData,CategoricalFeatures,OtherFeatures=['phone number'],title='Automated Exploratory Data Analysis for Churn Prediction')
# eda.TargetAnalysis('churn')
def find_correlation(df, thresh=0.9):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove
    params:
    - df : pd.DataFrame
    - thresh : correlation threshold, will remove one of pairs of features with
               a correlation greater than this value
    """
    
    corrMatrix = np.abs(df.corr())
    corrMatrix.loc[:,:] =  np.tril(corrMatrix, k=-1)

    already_in = set()
    result = []

    for col in corrMatrix:
        perfect_corr = corrMatrix[col][corrMatrix[col] > thresh].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)


    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat
telecomData["avg day call duration"]=telecomData['total day minutes']/telecomData['total day calls']
telecomData["avg eve call duration"]=telecomData['total eve minutes']/telecomData['total eve calls']
telecomData["avg night call duration"]=telecomData['total night minutes']/telecomData['total night calls']
telecomData["avg int call duration"]=telecomData['total intl minutes']/telecomData['total intl calls']

telecomData['Avg spent per call local']=telecomData[['total day charge','total night charge','total eve charge']].sum(axis=1)/telecomData[['total day calls','total night calls','total eve calls']].sum(axis=1)
telecomData['Avg spent per call intl']=telecomData['total intl charge']/telecomData['total intl calls']
feature_engineered_variables=telecomData.columns[-6:]
print("Engineered Features : ",", ".join(feature_engineered_variables))
telecomData[feature_engineered_variables].corr()
correlated_features=find_correlation(telecomData[feature_engineered_variables],thresh=0.90)
print(correlated_features)
numericalVariables.extend(feature_engineered_variables)
cols=[i for i in numericalVariables if i not in ['total day charge','total eve charge',
                                                'total night charge','total eve charge',
                                                 'avg int call duration'
                                                ]]
telecomData[cols].isnull().sum()[telecomData[cols].isnull().sum()>0]
telecomData.fillna(0,inplace=True)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

rfe = RFE(logreg)
rfe = rfe.fit(telecomData[cols], telecomData[targetVariable])

reduced_features=[cols[ix] for ix,i in enumerate(rfe.support_) if i==True]

print(reduced_features)
reduced_features.append(targetVariable)
# from dominance_analysis import Dominance

# dominance_classification=Dominance(data=telecomData[reduced_features],
#                     target=targetVariable,objective=0,pseudo_r2="mcfadden",top_k=len(reduced_features)-1)
# incr_variable_rsquare=dominance_classification.incremental_rsquare()
# dominance_classification.plot_incremental_rsquare()
# dominance_classification.dominance_stats()
# telecomData[targetVariable].astype(float)
reduced_features.remove(targetVariable)
import statsmodels.api as sm
logit_model=sm.Logit(telecomData[targetVariable].astype(float),telecomData[reduced_features].astype(float))
result=logit_model.fit()
print(result.summary())
print(categoricalVariables)
# international_map=(telecomData['international plan'].value_counts()/telecomData.shape[0]).to_dict()
# voice_mail_map=(telecomData['voice mail plan'].value_counts()/telecomData.shape[0]).to_dict()
# state_map=(telecomData['state'].value_counts()/telecomData.shape[0]).to_dict()
# internation_map,voice_mail_map,state_map
temp_df=telecomData.pivot_table(index='international plan',
                                columns=[targetVariable],
                                values=uniqueIdentifier,aggfunc=len)
temp_df.columns=['False','True']
temp_df['sum']=temp_df.sum(axis=1)
temp_df['churn_percentage']=temp_df['True']*100/temp_df['sum']
international_map=temp_df['churn_percentage'].to_dict()

international_map
temp_df=telecomData.pivot_table(index='state',
                                columns=[targetVariable],
                                values=uniqueIdentifier,aggfunc=len)
temp_df.columns=['False','True']
temp_df['sum']=temp_df.sum(axis=1)
temp_df['churn_percentage']=temp_df['True']*100/temp_df['sum']
state_map=temp_df['churn_percentage'].to_dict()

# state_map
temp_df=telecomData.pivot_table(index='voice mail plan',
                                columns=[targetVariable],
                                values=uniqueIdentifier,aggfunc=len)
temp_df.columns=['False','True']
temp_df['sum']=temp_df.sum(axis=1)
temp_df['churn_percentage']=temp_df['True']*100/temp_df['sum']
voice_mail_map=temp_df['churn_percentage'].to_dict()

voice_mail_map
def stateWiseChurnBucket(churn_rate):
    """
    * churn_rate <14.5 => 'normal'
    * churn_rate > 14.5 and  chrun_rate <= 17  => 'Medium'
    * churn_rate > 17  and churn_rate <= 19    =>  'High'
    * churn_rate > 19 and churn_rate <= 23     => 'Very-High'
    * churn_rate > 23 => 'Extremely-High'
    """
    if(churn_rate<=14.5):
        return 'Normal'
    elif((churn_rate>14.5) and (churn_rate<=17)):
        return "Medium"
    elif((churn_rate>17) and (churn_rate<=19)):
        return "High"
    elif((churn_rate>19) and (churn_rate<=23)):
        return "Very-High"
    else:
        return "Extremely-High"
    
    
stateWiseChurnBucket(14.5)
telecomData[telecomData['state'].isnull()]
# state_map
telecomData['state_map']=telecomData['state'].map(state_map)
telecomData['international_map']=telecomData['international plan'].map(international_map)
telecomData['voice_mail_map']=telecomData['voice mail plan'].map(voice_mail_map)
telecomData.head()
reduced_features.extend(telecomData.columns[-3:])
reduced_features
# stateWiseChurnBucket(14.5)
# telecomData['state_map'].max()
telecomData['state_bucket']=telecomData['state_map'].apply(stateWiseChurnBucket)
telecomData['state_bucket'].value_counts()


df=pd.get_dummies(telecomData['state_bucket'])
telecomData=pd.concat([telecomData,df],axis=1)

reduced_features.extend(telecomData.columns[-5:])
# temp_df=telecomData.pivot_table(index='state',
#                                 columns=[targetVariable],
#                                 values=uniqueIdentifier,aggfunc=len)
# temp_df.columns=['False','True']
# temp_df['sum']=temp_df.sum(axis=1)
# temp_df['churn_percentage']=temp_df['True']*100/temp_df['sum']
# temp_df['churn_percentage'].to_dict()
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

Train_x,Valid_x,train_y,valid_y=train_test_split(telecomData[reduced_features],
                                                 telecomData[targetVariable],
                                                 stratify=telecomData[targetVariable],
                                                 random_state=2020,test_size=0.3)

lgClf=LogisticRegressionCV(random_state=2020)

lgClf.fit(Train_x,train_y)
print(classification_report(train_y,lgClf.predict(Train_x)))
print(classification_report(valid_y,lgClf.predict(Valid_x)))
treeClf=DecisionTreeClassifier(
    random_state=42,
    min_impurity_decrease=10e-6,
    max_depth=3,
    class_weight={0:0.58473684, 1:3.45031056},
)

treeClf.fit(Train_x,train_y)
# treeClf.predict(Train_x)
print(classification_report(train_y,treeClf.predict(Train_x)))
print(classification_report(valid_y,treeClf.predict(Valid_x)))
# telecomData.shape[0] / (2 * np.bincount(telecomData[targetVariable]))
# telecomData[targetVariable].unique()
from sklearn import tree

# tree.plot_tree(treeClf)

tree.export_graphviz(treeClf,
                     out_file="tree.dot",
                     feature_names = reduced_features, 
                     class_names=['False','True'])

!dot -Tpng tree.dot -o tree.png
import os
os.listdir()
rfClf=RandomForestClassifier(random_state=2020,
                             n_estimators=500,
                             min_impurity_decrease=10e-5,max_depth=3
                             ,oob_score=True,
                             class_weight={0:0.58473684, 1:3.45031056},
                            )

rfClf.fit(Train_x,train_y)
print(classification_report(train_y,rfClf.predict(Train_x)))
print(classification_report(valid_y,rfClf.predict(Valid_x)))
gbClf=GradientBoostingClassifier(random_state=42,
     n_estimators=2500,
     learning_rate=0.01,
     min_impurity_decrease=10e-4,
     max_depth=3,
    min_samples_split=20
)

gbClf.fit(Train_x,train_y)
print(classification_report(train_y,gbClf.predict(Train_x)))
print(classification_report(valid_y,gbClf.predict(Valid_x)))
from imblearn.over_sampling import SMOTE
from collections import Counter
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(Train_x, train_y)
print('Resampled dataset shape %s' % Counter(y_res))


rfClf=RandomForestClassifier(random_state=42,
                             n_estimators=500,
                             min_impurity_decrease=10e-3
                             ,oob_score=True,
#                              class_weight={0:0.58473684, 1:3.45031056},
                            )

rfClf.fit(X_res,y_res)

print(classification_report(train_y,rfClf.predict(Train_x)))
print(classification_report(valid_y,rfClf.predict(Valid_x)))
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_score = rfClf.predict_proba(Valid_x)[:,1]

def plot_roc_curve(true_labels,predicted_probs):
    false_positive_rate, true_positive_rate, threshold = roc_curve(valid_y, y_score)
    print('roc_auc_score for Model : ', roc_auc_score(valid_y, y_score))

    plt.subplots(1, figsize=(10,10))
    plt.title('Receiver Operating Characteristic - Model ')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
plot_roc_curve(valid_y,y_score)

