# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import itertools

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

#train_data.head(10)
train_data["Name"] = train_data["Name"].str.split(',').str[1]

train_data["Name"] = train_data["Name"].str.split('.').str[0]

train_data["Name"] = train_data["Name"].str.strip()

x = train_data.groupby('Name').agg(['count']).index.get_level_values('Name')

x
train_data["Age"] = train_data.groupby("Name").transform(lambda x: x.fillna(x.mean()))['Age']

#changing sex to be 0 or 1 for female & male

train_data['Sex'].replace({'female':0,'male':1},inplace=True)

train_data.head()
train_data_tree = train_data.iloc[:,[False,False,True, False,True,True,True,True,False,True,False,False]]

train_labels_tree = train_data.iloc[:,1]

train_data_tree.describe()
#Need to create dummy variable columns for the Pclass variable. The other variables are either binary or numeric

train_data_tree_dummy = pd.concat([train_data_tree,pd.get_dummies(train_data_tree['Pclass'], prefix='Pclass')],axis=1)



train_data_tree_dummy.drop(["Pclass"],axis=1,inplace=True)

sib_sp = pd.cut(train_data_tree_dummy["SibSp"], 3,labels=[0,1,2]).tolist()

parch = pd.cut(train_data_tree_dummy["Parch"], 3,labels=[0,1,2]).tolist()



train_data_tree_dummy.drop(["Parch"],axis=1,inplace=True)

train_data_tree_dummy["SibSp2"] = np.where(train_data_tree_dummy.SibSp==0,0,1)

train_data_tree_dummy.drop(["SibSp"],axis=1,inplace=True)







train_data_tree_dummy.describe()
def gini_calc(data,labels):

    best_gini = []

    numeric_best_split = []

    numeric_indices = []

    counter = -1

    zeros = np.zeros(len(labels))

    ones = zeros + 1 

    zero_zero = 0

    zero_one = 0

    one_zero = 0

    one_one = 0

    gini_one = 0

    gini_two = 0

    weighted_gini = 0

    

    for i in data:

        counter +=1

        if data[i].dtype == 'int64' or data[i].dtype == 'uint8':

            list2 = []

            zero_zero = sum(np.logical_and(data[i] == labels, zeros == data[i])) #get counts of different combinations

            zero_one = sum(np.logical_and(data[i] != labels, zeros == data[i]))

            one_zero = sum(np.logical_and(data[i] != labels, ones == data[i]))

            one_one = sum(np.logical_and(data[i] == labels, ones == data[i]))



            if (one_one+one_zero) == 0:

                part_one = 0

                part_two = 0

            else:

                part_one = one_one/(one_one+one_zero)

                part_two = one_zero/(one_one+one_zero)

                

            if (zero_zero+zero_one) == 0:

                part_three = 0

                part_four = 0

            else:

                part_three = zero_zero/(zero_zero+zero_one)

                part_four = zero_one/(zero_one+zero_zero)



                

            gini_one = 1 - (part_one**2) - (part_two**2)

            gini_two = 1 - (part_three**2) - (part_four**2)

            weighted_gini = (gini_one * (one_one + one_zero)/(len(labels))) + (gini_two * (zero_zero + zero_one)/(len(labels)))

        

            best_gini.append(weighted_gini)



        else:

            

            numeric_indices.append(counter)

            numeric_vals = np.array(data.sort_values([i])[i].reset_index(drop=True))

            index_min = 0

            avg_lst = []

            numeric_gini_lst = []

            for h in range(len(numeric_vals)-1):

                avg_lst.append((numeric_vals[h]+numeric_vals[h+1])/2)

            zeros = np.zeros(len(labels))

            ones = zeros + 1 

            

            for val in avg_lst:

                vals_array = ones*val



                zero_zero = sum(np.logical_and(data[i] <= vals_array, zeros == labels)) #get counts of different combinations

                zero_one = sum(np.logical_and(data[i] <= vals_array, ones == labels))

                one_zero = sum(np.logical_and(data[i] >= vals_array, zeros == labels))

                one_one = sum(np.logical_and(data[i] >= vals_array, ones == labels))



                if (one_one+one_zero) == 0:

                    part_one = 0

                    part_two = 0

                else:

                    part_one = one_one/(one_one+one_zero)

                    part_two = one_zero/(one_one+one_zero)



                if (zero_zero+zero_one) == 0:

                    part_three = 0

                    part_four = 0

                else:

                    part_three = zero_zero/(zero_zero+zero_one)

                    part_four = zero_one/(zero_one+zero_zero)

                

                gini_one = 1 - (part_one**2) - (part_two**2)

                gini_two = 1 - (part_three**2) - (part_four**2)

                    

                weighted_gini = (gini_one * (one_one + one_zero)/(len(labels))) + (gini_two * (zero_zero + zero_one)/(len(labels))) 

                

                

                

                numeric_gini_lst.append(weighted_gini)

            index_min = np.argmin(numeric_gini_lst)

            numeric_best_split.append(numeric_vals[index_min])

            best_gini.append(min(numeric_gini_lst))

        

    return best_gini, numeric_best_split, numeric_indices

            

ginis, numeric_splits, numeric_index = gini_calc(train_data_tree_dummy,train_labels_tree)

ginis
def splitter(gini_array, data,labels, numeric_splits, numeric_index):

    """Splits a dataframe based on the best gini value determined from the gini function"""

    

    best_split = np.argmin(gini_array)

    best_gini = min(gini_array)

    best_var = data.columns[best_split]

    is_numeric = False

    tester = 0

    numeric_splitter = "Null"

    

    for i in numeric_index:

        if best_split == i:

            tester +=1

    

    if tester == 1:

        is_numeric = True

        numeric_splitter = numeric_splits[best_split]

        splitter = numeric_splits[best_split]

        combined = pd.concat([data, labels], axis=1, sort=False)

        df1 = combined[combined.iloc[:,best_split] <= splitter]

        df2 = combined[combined.iloc[:,best_split] > splitter]

        df1 = df1.drop(df1.columns[best_split], axis=1)

        df2 = df2.drop(df2.columns[best_split], axis=1)

        

        

        df1_labels = df1.iloc[:,-1]

        df1 = df1.iloc[:,:-1]



        df2_labels = df2.iloc[:,-1]

        df2 = df2.iloc[:,:-1]

        majority_class = 0

        if sum(df1_labels) > sum(df2_labels):

            majority_class = 1

        else:

            majority_class = 0

      

    else:

        combined = pd.concat([data, labels], axis=1, sort=False)



        df1 = combined[combined.iloc[:,best_split] == 0]

        df2 = combined[combined.iloc[:,best_split] == 1]

        df1 = df1.drop(df1.columns[best_split], axis=1)

        df2 = df2.drop(df2.columns[best_split], axis=1)   





        df1_labels = df1.iloc[:,-1]

        df1 = df1.iloc[:,:-1]



        df2_labels = df2.iloc[:,-1]

        df2 = df2.iloc[:,:-1]

        

        if sum(df1_labels) > sum(df2_labels):

            majority_class = 1

        else:

            majority_class = 0

    



    return df1, df2, df1_labels, df2_labels, best_gini, best_var, is_numeric, numeric_splitter, majority_class
def decision_tree(data,labels):

    """Takes a dataframe and labels and applied the Gini and splitter functions to it"""

    ginis, numeric_splits, numeric_index = gini_calc(data,labels)

    df_1, df_2, labs_one, labs_two, best_gini, best_var, is_numeric, numeric_splitter, majority_class = splitter(ginis,data,labels, numeric_splits, numeric_index)

    



    return best_gini, df_1, df_2, labs_one, labs_two, best_var, is_numeric, numeric_splitter, majority_class





best_gini, df_1, df_2, labs_one, labs_two, best_var, is_numeric, numeric_splitter,majority_class = decision_tree(train_data_tree_dummy,train_labels_tree)



def recursive_tree(data,labels,max_depth = 5):

    """Function that takes original data and labels and iteratively splits each dataframe for the full max-depth"""

    

    best_gini, df_1, df_2, labs_one, labs_two, best_var, is_numeric, numeric_splitter, majority_class = decision_tree(data,labels)

    

    data_frame_splits = []

    ginis = []

    labs_list = []

    best_var_list = []

    is_numeric_lst = []

    numer_splitter_lst = []

    majority_class_list = []

    data_frame_splits.append(data)

    ginis.append(.5)

    labs_list.append(labels)

    best_var_list.append("NA")

    is_numeric_lst.append("NA")

    numer_splitter_lst.append("NA")

    majority_class_list.append("NA")

    

    counter = 0

    for i in range(max_depth*2):

        if counter == 0:

            best_gini, df_1, df_2, labs_one, labs_two, best_var, is_numeric, numeric_splitter, majority_class = decision_tree(data,labels)

        else:

            best_gini, df_1, df_2, labs_one, labs_two, best_var, is_numeric, numeric_splitter,majority_class = decision_tree(data_frame_splits[counter],labs_list[counter])

        counter +=1

        data_frame_splits.append(df_1)

        data_frame_splits.append(df_2)

        ginis.append(best_gini)

        ginis.append(best_gini)

        labs_list.append(labs_one)

        labs_list.append(labs_two)

        best_var_list.append(best_var)

        best_var_list.append(best_var)

        is_numeric_lst.append(is_numeric)

        is_numeric_lst.append(is_numeric)

        numer_splitter_lst.append(numeric_splitter)

        numer_splitter_lst.append(numeric_splitter)

        majority_class_list.append(majority_class)

        if majority_class == 1:

            majority_class_list.append(majority_class-1)

        else:

            majority_class_list.append(majority_class+1)

        

    return data_frame_splits, ginis,labs_list,best_var_list, is_numeric_lst,numer_splitter_lst,majority_class_list



data_frame_splits, ginis,labs_list,best_var_list, is_numeric_lst,numer_splitter_lst,majority_class_list = recursive_tree(train_data_tree_dummy,train_labels_tree)
def create_rules(ginis, is_numeric_lst):

    """Returns indices of best tree as well as relavent information indexed by these best indices"""

    counter = 0

    include_list = []

    for i in range(len(ginis)):

        if i == 0:

            include_list.append(counter)

        

        elif is_numeric_lst[i] == False and counter % 2 != 0:

            if ginis[i] <= ginis[int((counter-1)/2)]:

                include_list.append(counter)

        elif is_numeric_lst[i] == False and counter % 2 == 0:

            if ginis[i] <= ginis[int((counter-2)/2)]:

                 include_list.append(counter)

        elif is_numeric_lst[i] == True and counter % 2 != 0:

            if ginis[i] <= ginis[int((counter-1)/2)]:

                include_list.append(counter)

        elif is_numeric_lst[i] == True and counter % 2 == 0:

            if ginis[i] <= ginis[int((counter-2)/2)]:

                include_list.append(counter)

        counter +=1

    

    best_variables = []

    for i in include_list:

        best_variables.append(best_var_list[i])

    

    is_numeric_final = []

    for i in include_list:

        is_numeric_final.append(is_numeric_lst[i])

    

    majority_class_list_final = []

    for i in include_list:

        majority_class_list_final.append(majority_class_list[i])

    

    numer_splitter_lst_final = []

    for i in include_list:

        numer_splitter_lst_final.append(numer_splitter_lst[i])

    

    

    return include_list, is_numeric_final, best_variables, majority_class_list_final, numer_splitter_lst_final

    

inclusions, numeric_final, best_vars, majority_class_lst, numer_splitter_lst_final = create_rules(ginis, is_numeric_lst)

print(inclusions)
def yarf(data,is_numeric_final,best_variables, inclusions,majority_class_list_final,numer_splitter_lst_final, counter):

    """yet another recursive function... go through the data and set a column equal to a value determined by the rules"""

    if is_numeric_final[counter] == False:

        x = data[data[best_variables[counter]] == 0].copy()

        if majority_class_list_final[counter] == 1:

            x["Survived"] = 0

        elif majority_class_list_final[counter] == 0:

            x["Survived"] = 1

            

        y = data[data[best_variables[counter]] == 1].copy()

        if majority_class_list_final[counter] == 0:

            y["Survived"] = 0

        elif majority_class_list_final[counter] == 1:

            y["Survived"] = 1

        return x, y

        

    elif is_numeric_final[counter] == True:

        x = data[data[best_variables[counter]] <= numer_splitter_lst_final[counter]].copy()

        if majority_class_list_final[counter] == 1:

            x["Survived"] = 0

        elif majority_class_list_final[counter] == 0:

            x["Survived"] = 1

        y = data[data[best_variables[counter]] > numer_splitter_lst_final[counter]].copy()

        if majority_class_list_final[counter] == 0:

            y["Survived"] = 0

        elif majority_class_list_final[counter] == 1:

            y["Survived"] = 1

        return x, y

    



def recursive_through_yarf(data,is_numeric_final,best_variables, inclusions,majority_class_list_final,numer_splitter_lst_final, max_depth = 5):

    """Using the yarf function, iteratively save the dataframes with the predictions as a column"""

    x, y = yarf(data,is_numeric_final,best_variables, inclusions,majority_class_list_final,numer_splitter_lst_final,1)

    

    list_of_data_frames = []

    list_of_data_frames.append(x)

    list_of_data_frames.append(y)

    counter = 2

    frames_counter = 0

    

    for i in range(max_depth):

        x, y = yarf(list_of_data_frames[frames_counter],is_numeric_final,best_variables, inclusions,majority_class_list_final,numer_splitter_lst_final,counter)

        list_of_data_frames.append(x)

        list_of_data_frames.append(y)

        counter+=1

        frames_counter +=1

        x, y = yarf(list_of_data_frames[frames_counter],is_numeric_final,best_variables, inclusions,majority_class_list_final,numer_splitter_lst_final,counter)

        list_of_data_frames.append(x)

        list_of_data_frames.append(y)

        counter+=1

        frames_counter +=1

    

    return list_of_data_frames





#testing = recursive_through_yarf(train_data_tree_dummy,is_numeric_final,best_variables, inclusions,majority_class_list_final,numer_splitter_lst_final,max_depth = 3)

    

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data["Name"] = test_data["Name"].str.split(',').str[1]

test_data["Name"] = test_data["Name"].str.split('.').str[0]

test_data["Name"] = test_data["Name"].str.strip()

test_data["Age"] = test_data.groupby("Name").transform(lambda x: x.fillna(x.mean()))['Age']

#changing sex to be 0 or 1 for female & male

test_data['Sex'].replace({'female':0,'male':1},inplace=True)

test_data_tree = test_data.iloc[:,[False,True,False, True,True,True,True,False,True,False,False]]

test_data_tree.head()



test_data_tree_dummy = pd.concat([test_data_tree,pd.get_dummies(test_data_tree['Pclass'], prefix='Pclass')],axis=1)



test_data_tree_dummy.drop(["Pclass"],axis=1,inplace=True)

sib_sp = pd.cut(test_data_tree_dummy["SibSp"], 3,labels=[0,1,2]).tolist()

parch = pd.cut(test_data_tree_dummy["Parch"], 3,labels=[0,1,2]).tolist()

test_data_tree_dummy["SibSp2"] = np.where(test_data_tree_dummy.SibSp==0,0,1)



test_data_tree_dummy.drop(["Parch"],axis=1,inplace=True)

test_data_tree_dummy.drop(["SibSp"],axis=1,inplace=True)

test_data_tree_dummy.head()

testing = recursive_through_yarf(test_data_tree_dummy,numeric_final,best_vars, inclusions,majority_class_lst,numer_splitter_lst_final,max_depth = 5)

def make_predictions(testing_data, inclusions):

    """When merging dataframes, only want to merge leaf nodes. This function takes all the leaf nodes and merges together"""

    

    for i in inclusions:

        if i %2 !=0:

            try:

                inclusions.remove(int((i-1)/2))

            except:

                pass

        elif i %2 ==0:

            try:

                inclusions.remove(int((i-2)/2))

            except:

                pass

    inclusions2 = []

    for i in inclusions:

        inclusions2.append(i-1)

        

    dataframes_to_keep = []

    for i in inclusions2:

        dataframes_to_keep.append(testing_data[i])





    preds = pd.concat(dataframes_to_keep, axis=0).sort_index(axis = 0)



    return preds

    

preds = make_predictions(testing, inclusions)
preds.shape
test_data.shape
test_data.head()
data = {'PassengerId': test_data["PassengerId"].values, 'Survived':preds["Survived"].values}



df_submission = pd.DataFrame(data)



df_submission.to_csv("submission_decision_tree.csv",index=False)