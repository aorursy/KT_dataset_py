import warnings

warnings.filterwarnings('ignore') # Disabling warnimgs for clearer outputs



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





import category_encoders as ce        
#Load TRAIN data

train =     pd.read_csv("/kaggle/input/titanic/train.csv")

#Load TEST data

test =      pd.read_csv("/kaggle/input/titanic/test.csv")



train.head()
all_data=pd.concat([train, test]).reset_index(drop=True)

all_data
def all_die_model(data):

    

    data["Survived"][(data["PassengerId"] > 891)] = 0

    

    return data



def make_submission (data,file_name):    

    

    data["Survived"] = data["Survived"].astype(int)

    data.loc[891:, ["PassengerId", "Survived"]].to_csv('/kaggle/working/'+file_name, index=False)

    

    

    
all_data=all_die_model(all_data)
#make All Die Model submission

make_submission(all_data,"all_die_model.csv")
def gender_model(data):

    # make gender model

    data["Survived"][(data["PassengerId"] > 891)] = 0

    data["Survived"][(data["PassengerId"] > 891) & (data["Sex"] == 'female')] = 1

    

    return data
all_data=gender_model(all_data)
#make our gender submission

make_submission(all_data,"gender_model.csv")

all_data
def woman_childs_model(data):

    

    #ADD need features

    data["Last_Name"]=data.Name.str.split(',').str[0]

    data["Boy"] = ((data.Name.str.contains('Master', case=False) == True) | ((data.Sex=='male') & (data.Age<14))).astype('int')

    data["Sex2"]=data["Sex"]

    data["Sex2"][(data["Boy"]==1) | (data["Age"]<14)]="children"

    

    #Group_LPE Last_Name_Pclass_Embarked

    data["Group_LPE"] = data["Last_Name"].astype(str) + "_" + data["Pclass"].astype(str)+ "_" + data["Embarked"].astype(str)  



 



    #ADD to group Group_LPE persons with same tickets number

    data["Group_Big"]=data["Group_LPE"]

    for index, row in data[data["Group_LPE"] != "NO"].groupby("Group_LPE"):

        for ticket in row.Ticket:        

            data["Group_Big"][(data["Ticket"]==ticket) & (data["Group_LPE"]!=index)]= index

    

    #ADD to group Group_LPE persons with same CABIN

    for index, row in data[data["Group_Big"] != "NO"].groupby("Group_Big"):

        for Cabin in row.Cabin:

            data["Group_Big"][(data["Cabin"]==Cabin) & (data["Group_Big"]!=index) & (Cabin!="Unknown")]= index

    



    

    

    

    #delete all adults mens from Group_WC

    data["Group_WC"]=data["Group_Big"]

    data["Group_WC"][data["Sex2"]=='male']="NO"     

    

    

    

    data["Group_Survived"] = 0

    data["Group_Survived"][data['Group_WC'] != "NO"] = 10



    for index, row in data.iterrows():

        # print(index,row)

        if row['Group_WC'] != "NO":

            # print(index, row['Group_ID'], " - ", row["Group_Count"], " - ", row["Survived"])



            if (row["Survived"] == 0) & (row["Pclass"] == 3):

                data["Group_Survived"][data['Group_WC'] == row['Group_WC']] = 0



            if row["Survived"] == 1:

                if data["Group_Survived"][data['PassengerId'] == row['PassengerId']].item() > 0:

                    data["Group_Survived"][data['Group_WC'] == row['Group_WC']] = 1



    data["Group_Survived"][(data['Group_Survived'] == 10) & (data['Pclass'] < 3)] = 1

    data["Group_Survived"][(data['Group_Survived'] == 10) & (data['Pclass'] == 3)] = 0

    

    

    

    data['Group_WC_Count'] = data["Group_WC"]

    count_enc = ce.CountEncoder()

    count_encoded = count_enc.fit_transform(data["Group_WC_Count"])

    data['Group_WC_Count'] = count_encoded

    data["Group_WC"][data['Group_WC_Count'] == 1] = "NO" 



   

    # make gender model

    data["Survived"][(data["PassengerId"] > 891)] = 0 #ALL DIE MODEL

    data["Survived"][(data["PassengerId"] > 891) & (data["Sex"] == 'female')] = 1 #GENDER MODEL

    

    # add woman_childs_model

    data["Survived"][(data["PassengerId"] > 891) & (data["Group_WC"] != "NO") & (data["Group_Survived"] == 0)] = 0

    data["Survived"][(data["PassengerId"] > 891) & (data["Group_Survived"] == 1)] = 1





    return data    

    
all_data=pd.concat([train, test]).reset_index(drop=True)

all_data=woman_childs_model(all_data)

#make submission

make_submission(all_data,"woman_childs_model.csv")
male_alive=all_data[(all_data["PassengerId"] > 891) & (all_data["Sex"] == 'male') & (all_data["Survived"] == 1)]



display("All Males Alive=",len(male_alive))

display(male_alive[["PassengerId","Pclass","Name","Sex","Age","Cabin","Last_Name"]])



females_perish=all_data[(all_data["PassengerId"] > 891) & (all_data["Sex"] == 'female') & (all_data["Survived"] == 0)]



display("Females Perish=",len(females_perish))

display(females_perish[["PassengerId","Pclass","Name","Sex","Age","Cabin","Last_Name"]])
#All groups we can see here

all_data.groupby(["Group_WC"])[["Group_WC_Count"]].count() 
#Add New Features



all_data["Age_Missed"]=(all_data["Age"].isnull().astype(int))

all_data['Cabin'].fillna('Unknown', inplace=True)

all_data["Cabin_Missed"]=(all_data["Cabin"]=="Unknown").astype(int)

all_data['Name_count']=all_data["Name"].apply(lambda x: len(x.split())).astype(int)



#Make model



all_data["Survived"][(all_data["Sex"] == 'female') & (all_data["Pclass"] == 3) & (all_data["Age_Missed"] ==1)]  = 0

all_data["Survived"][(all_data["Sex"] == 'female') & (all_data["Pclass"] == 3) & (all_data["Name_count"] ==3)]  = 0

all_data["Survived"][(all_data["Sex"] == 'male')   & (all_data["Pclass"] == 3) & (all_data["Cabin_Missed"]==0)] = 1
#make less information submission

make_submission(all_data,"less_information_model.csv")