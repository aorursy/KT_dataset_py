



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("../input/train.csv")





import numpy as np

from sklearn.ensemble import RandomForestClassifier  # Mike Bernico used RandomForestRegressor

from sklearn.metrics import roc_auc_score



y=df.pop("Survived")

model=RandomForestClassifier(n_estimators=200,max_leaf_nodes=6,max_features='auto', oob_score=True,n_jobs=-1,random_state=0)

# Mike used n_estimator of 1000; my optimization is 200.

def preprocessing(df):

    df['Age'].fillna(df.Age.mean(),inplace=True)

    # convert Cabin to number4s

    def clean_Cabin(x):

        try:

            return ord(x[0])

        except:

            return 300  # special case give arbitrary number

    df['Cabin']=df.Cabin.apply(clean_Cabin)

    

    #convert sex to 0/1

    def convert_sex(x):

        if (str.upper(x)=='MALE'):

            return 0

        else:

            return 1

    df['Sex']=df.Sex.apply(convert_sex)

    

    #convert Embarked to numbers, except assign max values"C", othervalues assign 3

    def convert_Embarked(x):

        try:

            if(str.upper(x)=='S'):

                return 0

            elif(str.upper(x)=='C'):

                return 1

            elif(str.upper(x)=='Q'):

                return 2

            else:

                return 3

        except:

            return 1

    def convert_nan(x):

        try:

            return float(x)

        except:

            return 0

    df['Fare'].apply(convert_nan)

    df['Embarked']=df.Embarked.apply(convert_Embarked)

    df['family_size']=df.Parch+df.SibSp+1  #create new feature

    

    #extract ticket number, it shows negative effect

    def convert_ticket(x):

        try:

            a=x.split(' ')

            n=len(a)

            #print(a)

            if(n>1):

                return int(a[-1])

            else:

                return int(a[0])

                print(a)

        except:

            return 0

    

    #if there is additional words before ticket number, 1, otherwise 0; negative effect

    def convert_ticket2(x):

        a=x.split(' ')

        n=len(a)

        if(n>1):

            return 1

        else:

            return 0

        

                

    #df['tic']=df.Ticket.apply(convert_ticket)

   # df['tic_speccial']=df.Ticket.apply(convert_ticket2)

    df.drop(['Name','Ticket','PassengerId'],axis=1,inplace=True)

    return df



df=preprocessing(df)



model.fit(df.values,y)



#in test data, tehre is null values

def check_null(x):

    if pd.isnull(x):

        print(x)

test=pd.read_csv('../input/test.csv')

test=preprocessing(test)

#print(pd.isnull(test).sum()) # summary of null values

test['Fare']=test['Fare'].fillna(0) # fill it with 0

test.Fare.apply(check_null)

pred=model.predict(test.values[:,:])



y2=pd.read_csv('../input/gendermodel.csv').pop('Survived')

print("accuracy for test data:",roc_auc_score(y2,pred))

re=pd.DataFrame(np.array([y2,pred]).T)

import matplotlib.pyplot as plt

plt.plot(re[0]-re[1]) # this shows the differences between actual and prediction; 