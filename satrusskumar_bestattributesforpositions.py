import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
def heightConv(height):
    feet=height.split("\'")[0]
    inch=height.split("\'")[1]
    return((float(feet)*12)+float(inch))
df=pd.read_csv('./fifadata.csv')
df.drop(columns=['Unnamed: 0'],inplace=True)
poslst=['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
# We do not consider position list for predicting another position attribute, we need to consider physical and other atrributes
# Function for correlation and linear regression analysis for any position
def GetAttributeWeights(dframe,position, testsize):
    dframe=dframe.drop(columns=poslst)
    
    #print(len(dframe.columns))
    # Finding best position attributes by taking players of only that position
    dframe=dframe[dframe.Position==position].reset_index(drop=True)
    len(dframe)

    dframe.columns=dframe.columns.str.replace(' ','')
    dframe_attr=dframe[["Age","Overall","Potential","Special","InternationalReputation","WeakFoot","SkillMoves","WorkRate","Height","Weight",
           "Crossing","Finishing","HeadingAccuracy","ShortPassing","Volleys","Dribbling","Curve","FKAccuracy","LongPassing",
           "BallControl","Acceleration","SprintSpeed","Agility","Reactions","Balance","ShotPower","Jumping","Stamina","Strength","LongShots","Aggression","Interceptions","Positioning","Vision","Penalties","Composure","Marking","StandingTackle","SlidingTackle","GKDiving","GKHandling","GKKicking","GKPositioning","GKReflexes"]]

    # Convert height to inches (numerical) and weight (numerical)
    dframe_attr.Height=dframe_attr.Height.apply(lambda x : heightConv(x))
    dframe_attr.Weight=dframe_attr.Weight.apply(lambda x : float(str(x).replace('lbs','')))
    #print(dframe_attr.WorkRate.unique())
    #print(dframe_attr.SkillMoves.unique())
    # No unique work rate for players which can be dropped
    dframe_attr=dframe_attr.drop(columns=['WorkRate','SkillMoves'])
    for col in dframe_attr:
        dframe_attr[col]=dframe_attr[col].astype('float')
    #print(len(dframe_attr.columns))
    
    dframe_attr_corr=dframe_attr.corr()[['Overall']].reset_index().rename(columns={'index':'Attribute'}).sort_values('Overall',ascending=False).reset_index(drop=True)
    dframe_attr_corr=dframe_attr_corr[dframe_attr_corr.Attribute!='Overall']
    
    # Get regression weights
    X=dframe_attr.drop(columns=['Overall']).values
    y=dframe_attr[['Overall']].values

    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=testsize, random_state=18)
    lr=LinearRegression()
    lr.fit(X_train,y_train)
    print("Train accuracy", lr.score(X_train,y_train))
    print("Test accuracy", lr.score(X_test,y_test))

    # 99% accuracy even we decrease the train size which is highly linealy distributed data
    # Let us check weights of attributes
    dframe_attr_lr=pd.DataFrame(list(dframe_attr.drop(columns=['Overall']).columns),[lr.coef_[0]]).reset_index().rename(columns={'level_0':'Weight',0:'Attribute'})[['Attribute','Weight']].sort_values('Weight',ascending=False).reset_index(drop=True)
    return(dframe_attr_corr,dframe_attr_lr)
gk_cor,gk_lr=GetAttributeWeights(df,'GK',0.3)
print('Linear Regression')
print(gk_lr[:5])
print('\n')
print('Correlation')
print(gk_cor[:5])
st_cor,st_lr=GetAttributeWeights(df,'ST',0.3)
print('Linear Regression')
print(st_lr[:7])
print('\n')
print('Correlation')
print(st_cor[:7])