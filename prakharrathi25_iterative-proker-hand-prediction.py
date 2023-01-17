# Getting the test and train data 
!wget https://s3.eu-central-1.wasabisys.com/aicrowd-public-datasets/aicrowd_educational_pkhnd/data/public/test.csv
!wget https://s3.eu-central-1.wasabisys.com/aicrowd-public-datasets/aicrowd_educational_pkhnd/data/public/train.zip
!unzip train.zip
# Importing necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Machine Learning 
from sklearn.model_selection import train_test_split
train_data_path = "train.csv" #path where data is stored
train_data = pd.read_csv(train_data_path) #load data in dataframe using pandas
# Getting the test data 
test_data_path = "test.csv"
test_data = pd.read_csv(test_data_path)
train_data.shape, test_data.shape
# Let's look at the training data 
train_data.head()
# Data Labels 
train_data['label'].value_counts() 
train_data.head()
line = train_data.iloc[1]
print(type(line))
# Creating a testing dictionary 
test = {'S1': 1,
        'C1': 2,
        'S2': 1,
        'C2': 3,
        'S3': 2,
        'C3': 4,
        'S4': 2,
        'C4': 5,
        'S5': 2,
        'C5': 6}
# FUNCTION TO EXTRACT DATA FROM A ROW 
def series_to_dict(ser):
    dic = {} 
    
    # Allot dictionary positions
    dic['S1'] = ser[0]
    dic['C1'] = ser[1]
    dic['S2'] = ser[2]
    dic['C2'] = ser[3]
    dic['S3'] = ser[4]
    dic['C3'] = ser[5]
    dic['S4'] = ser[6]
    dic['C4'] = ser[7]
    dic['S5'] = ser[8]
    dic['C5'] = ser[9]
    
    # Return the dectionary 
    return dic

# Working and tested 
line = train_data.iloc[1]
print(series_to_dict(line))

# Function to check for flush
def check_flush(dic):
    
    # Extract suites 
    S1 = dic['S1'] 
    S2 = dic['S2'] 
    S3 = dic['S3'] 
    S4 = dic['S4'] 
    S5 = dic['S5'] 
    
    # Check if all suites same 
    if S1 == S2 == S3 == S4 == S5:
        return 1    # all matching suites
    else: 
        return 0

# Working and tested 
# Function to check for Straight 
def check_straight(dic):
    
    '''
      input: dictionary containing all cards' info
      
    '''
    
    # Extract classes
    C1 = dic['C1'] 
    C2 = dic['C2'] 
    C3 = dic['C3'] 
    C4 = dic['C4'] 
    C5 = dic['C5']
    
    class_flag = 1
    
    # Make a class list 
    C = [C1, C2, C3, C4, C5]
    
    # Sort the list 
    C.sort()
    
    # Start checking for a 
    for i in range(len(C)-1):
        if C[i+1]==C[i]+1:
            pass
        else:
            class_flag = 0
            break
    return class_flag
    
# Tested and Working 
check_straight(test)
# Check remaining combinations
def check_from_4_to_9(dic):
    
    '''
      input: dictionary containing all cards' info
      
    '''
    
    # Extract classes
    C1 = dic['C1'] 
    C2 = dic['C2'] 
    C3 = dic['C3'] 
    C4 = dic['C4'] 
    C5 = dic['C5']
    
    # Counts the number of unique cards in the sorted list
    counter = 1
    
    # Make a class list 
    C = [C1, C2, C3, C4, C5]
    C.sort()
    
    # Looping through all cases
    for i in range(1,5):
        if C[i] == C[i-1]:
            pass
        else: 
            counter += 1
    
    # 4 unique cards mean one pair
    if counter == 4:
        return 1    # Assigned label 
    
    # 3 unique cards could mean 2 pair or 3 of a kind 
    elif counter == 3: 
        
        # create counters for val counts 
        count = [1, 0, 0]
        
        j = 0
        for i in range(1, 5):
            if C[i] == C[i-1]:
                count[j] += 1
            else:
                j+=1
                count[j] += 1
        # Sort value counts list 
        count.sort()
        
        if count[2] == 2:
            return 2    # TWO PAIR CONDITION SATISFIED 
        
        if count[2] == 3: 
            return 3    # THREE OF A KIND CONDTITION SATISFIED 
    
    # If 2 unique cards then we could have 4 of a kind or full house 
    elif counter == 2: 
        
        # Check condition for four of a kind 
        if C[0] == C[1]:
            if C[0] == C[1] == C[2] == C[3]:
                return 7    # Four of a kind contion satisfied
            else:
                # Check full house conditions 
                return 6
        else:
            # Only four of a kind possible 
            return 7
    
    # If none
    else:
        return 0
                
# Tested and working      
            
check_from_4_to_9(test)
def check_royal(dic):
    
    '''
      input: dictionary containing all cards' info
      
    '''
    
    # Extract classes
    C1 = dic['C1'] 
    C2 = dic['C2'] 
    C3 = dic['C3'] 
    C4 = dic['C4'] 
    C5 = dic['C5']
    
    # Make a class list 
    C = [C1, C2, C3, C4, C5]
    C.sort()
    
    # Check for Royal 
    if C[0] == 1: 
        for i in range(1, 5):
            if C[i] != 9 + i:
                       return 0
        return 1
# Function to assign labels to the series data 
def assign_hand_label(ser):
    
    '''
      input: Series containing all the card information
      output: card hand label
      
    '''
    
    # Extract Data 
    hand = series_to_dict(ser)
    # Check if a flush 
    if check_flush(hand):
        
        # Check if a Royal Flush 
        if check_royal(hand):
            return 9
        
        # Check if straight flush 
        elif check_straight(hand):
            return 8
        
        else: 
            return 5 
    
    # Check for the rest 
    if check_straight(hand) or check_royal(hand):
        return 4
    else:
        return check_from_4_to_9(hand)
    
#     # If all fail 
#     else: 
#         return 0 
line=train_data.iloc[815]
assign_hand_label(line)
pred = [] 
for i in range(len(train_data)):
    line = train_data.iloc[i]
    val = assign_hand_label(line)
    print("{}. True Value: {} and Predicted Value: {}".format(i, line['label'], val))
    pred.append(val)


# Check accuracy 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_train = train_data['label'].values

# Convert pred to series
pred = pd.Series(pred)
acc_sc = accuracy_score(y_true=y_train, y_pred=pred)
print(acc_sc)
print(classification_report(y_true=y_train, y_pred=pred))
print(confusion_matrix(y_true=y_train, y_pred=pred))
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 6))
sns.heatmap(confusion_matrix(y_true=y_train, y_pred=pred), annot=True, cmap='Accent_r')
print(pred.unique())
test_pred = [] 
for i in range(len(test_data)):
    line = test_data.iloc[i]
    val = assign_hand_label(line)
    test_pred.append(val)

# sub = pd.DataFrame({'label' : test_pred})
# sub
# sub.to_csv('submission2.csv', index=False)