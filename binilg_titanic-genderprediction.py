import pandas as pd

Train = pd.read_csv("../input/train.csv")

print ("Total count = ",Train.shape[0])  #this will print the count of the rows

Women_Count = Train.loc[Train['Sex']=='female'].shape[0] #this will print the count of female passengers



print ("Woman Count = ",Women_Count)

Men_Count = Train.loc[Train['Sex']=='male'].shape[0]

Men_Survived = Train.loc[(Train["Sex"]=="male") & (Train["Survived"]==1)].shape[0]

Men_Survival_Rate = float(Men_Survived)/float(Men_Count)*100

Women_Survived = Train.loc[(Train["Sex"]=="female") & (Train["Survived"]==1)].shape[0]

Women_Survival_Rate = float(Women_Survived)/float(Women_Count)*100

if Women_Survival_Rate > Men_Survival_Rate:

    Women = True

else:

    Women = False

#Prediction part

Test = pd.read_csv("../input/test.csv")

Test.head()

print (Test.shape[0])

headers = ["Passengerid","Survived"]



PredictionFile = open("gender.csv","w")

PredictionFile.seek(0)

PredictionFile.truncate()

PredictionFile.write('\t'.join(headers)+'\n')

for i in range(0,len(Test)):

    if Test.iloc[i]['Sex'] == 'female':

        body = Test.iloc[i]['Name'],"1"

        PredictionFile.write('\t'.join(body)+'\n')

    else:

        body = Test.iloc[i]['Name'],"0"

        PredictionFile.write('\t'.join(body)+'\n')
()