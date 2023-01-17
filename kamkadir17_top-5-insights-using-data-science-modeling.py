import datetime

import calendar



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter



#from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import classification_report



from sklearn.utils import resample



import seaborn as sns



%matplotlib inline



np.random.seed(0)

fmt = lambda x,pos: '{:.0%}'.format(x)
# Collect Initial Data

df = pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv')

df.shape
# Sample & Explore Data

df.head()
# Describe Data

# Lets now infer the basic information of the dataframe and see what columns might be useful

df.info()



# based on the data - Initial findings / questions that come up are as follows

# There seems to be no NaN data. This total rows in the dataset is 110,527 and all columns have the same number of values.

# We do have two dates - One the day when the appointment was scheduled and the day when the actual appointment was.

# Probably we can derive some additional features from these dates to see how far back the appointment was made? Does it have any weight on missing the appointment ?

# We also have few categorical values - which we might need to do One-Hot encoding so that we have each unique value in the category represented in a binary format. Also we need only N-1 unique values to be represented in One-Hot encoding so we will be removing the excess dummy value.

# We do have Age. We can classify the Age into multiple groups to understand which Age group honors more the appointment and which doesn't. Like if the appointment is for kids, do they make it or if it was for senior people do they make it ?



# Lets start exploring and try and find answers to these intriguing questions.
#Verify Data Quality for NaN

#Looks like there are no missing values



df.dropna(how="any").shape
# Analysis 1.1

# A simple crosstab of Gender vs No-show 



pd.crosstab(df.Gender, df['No-show'], margins=True, margins_name="Total")
# Analysis 1.2

# Crosstab between Gender vs No-show normalized to percentages to identify its impact. 

# We then use Seaborn heatmap to visually see the output.





ax = sns.heatmap(pd.crosstab(df.Gender, df['No-show'], normalize='index'),

            cmap="YlGnBu", annot=True,  cbar=True, fmt='.0%',

            cbar_kws={'format': FuncFormatter(fmt)});

bottom, top = ax.get_ylim()

ax.set_title("Gender vs Medical No Show")

ax.set_ylim(bottom + 0.1, top - 0.1);

ax.get_figure().savefig('./gender_noshow.png')
# Analysis 1.3

# Crosstab between Gender vs No-show normalized to percentages to identify its impact. 

# We then use Seaborn heatmap to visually see the output.



ax = sns.heatmap(pd.crosstab(df.Gender, df['No-show'], normalize='columns'),

            cmap="YlGnBu", annot=True,  cbar=False, fmt='.0%',

            cbar_kws={'format': FuncFormatter(fmt)});

bottom, top = ax.get_ylim()

ax.set_title("Gender vs Medical No Show")

ax.set_ylim(bottom + 0.1, top - 0.1);

ax.get_figure().savefig('./gender_appoints_booked.png')
# To do this analysis, we need to first get the weekday of the appointment. We can do this by writing a helper lamba function



#1. Create two new columns for populating the weekday_number and 'day of the week' of the appointment day. 

# Its a numeric value - 0 to 6 representing Monday to Sunday.



wd_num = lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").date().weekday()

wd = lambda x: calendar.day_name[x]

df['weekday_num'] = df['AppointmentDay'].apply(wd_num)

df['weekday'] = df['weekday_num'].apply(wd)



df.tail()
# Analysis 2.1

#Crosstab between weekday and total appointments and No-show details



pd.crosstab([df.weekday_num, df.weekday], df['No-show'],rownames=["Weekday#","Day of the Week"], colnames=["No Show"], margins=True, margins_name="Total")
# Analysis 2.2

# Crosstab between Weekday vs No-show normalized to percentages to identify its impact. 

# We then use Seaborn heatmap to visually see the output.



ax = sns.heatmap(pd.crosstab([df.weekday_num, df.weekday], df['No-show'],rownames=["Weekday#","Day of the Week"], colnames=["No Show"], normalize='index'),

            cmap="YlGnBu", annot=True,  cbar=True, fmt='.0%',

            cbar_kws={'format': FuncFormatter(fmt)});

bottom, top = ax.get_ylim()

ax.set_title("Weekday vs Medical No Show")

ax.set_ylim(bottom + 0.1, top - 0.1);

ax.get_figure().savefig('./weekday_noshow.png',pad_inches=0.3)
# Analysis 3.1

# Crosstab between SMS vs No-show normalized to percentages to identify its impact. 



# Lets find out how many number of days before they booked the appointment. 

# And lets see if there is any trend in these bookings being missed / SMS helping not to miss the appointments



df['SMS_Desc'] = df['SMS_received'].apply(lambda x: 'Sent' if x == 1 \

                                                                   else 'Not Sent')



ax = sns.heatmap(pd.crosstab([df.SMS_Desc], df['No-show'],rownames=["SMS Reminder"], colnames=["No Show"],  normalize='index'),

            cmap="YlGnBu", annot=True,  cbar=True, fmt='.0%',

            cbar_kws={'format': FuncFormatter(fmt)});

bottom, top = ax.get_ylim()

ax.set_title("SMS Received vs Medical No Show")

ax.set_ylim(bottom + 0.1, top - 0.1);

ax.get_figure().savefig('./sms_received_noshow.png')
# Before we do this, lets add a couple of new features that might come in handy



#1. Lets also group the patients based on Age into 6 broad categories and see if any of these groups had any impact

#   Baby               -> 0 - 3

#   Kid                -> 4  - 12

#   Adolescent         -> 13 - 19

#   Young_Adult        -> 20 - 39

#   Adult              -> 40 - 64

#   Senior             -> 65 & Above
# Lets do a quick summary of value counts for each Age. We see that there is 1 record with Age = -1. 

# We will consider this age row also as a Baby for our analysis.

df.Age.value_counts()
def create_age_groups(Age):

    """

    This function creates a new feature to the input dataframe based on the Age.

    

    It creates categorical values based on Age. It categorizes the Age into 1 of the 6 categories.

    #   Baby               -> 0 - 3

    #   Kid                -> 4  - 12

    #   Adolescent         -> 13 - 19

    #   Young_Adult        -> 20 - 39

    #   Adult              -> 40 - 64

    #   Senior             -> 65 & Above

    

    INPUT:

    col - Age column

    

    OUTPUT:

    Age_Group

    

    """

    if Age >= 65:

        Age_Group = 'Senior'

    elif Age >= 40:

        Age_Group = 'Adult'

    elif Age >= 20:

        Age_Group = 'Young_Adult'

    elif Age >= 13:

        Age_Group = 'Adolescent'

    elif Age >= 4:

        Age_Group = 'Kid'   

    else:

        Age_Group = 'Baby'         

    

    return Age_Group



df['Age_Group'] = df['Age'].apply(create_age_groups)

df.tail()
# Analysis 4.1

# Crosstab between Age_Group vs No-show normalized to percentages to identify its impact. 

# We then use Seaborn heatmap to visually see the output.



ax = sns.heatmap(pd.crosstab([df.Age_Group], df['No-show'],rownames=["Age Group"], colnames=["No Show"], normalize='index'),

            cmap="YlGnBu", annot=True,  cbar=True, fmt='.0%',

            cbar_kws={'format': FuncFormatter(fmt)});

bottom, top = ax.get_ylim()

ax.set_title("Age Group vs Medical No Show %")

ax.set_ylim(bottom + 0.1, top - 0.1);

ax.get_figure().savefig('./age_group_noshow_percent.png');
# Analysis 4.2

# Crosstab between Age_Group vs No-show normalized to percentages to identify its impact. 

# We then use Seaborn heatmap to visually see the output.



ax = sns.heatmap(pd.crosstab([df.Age_Group], df['No-show'],rownames=["Age Group"], colnames=["No Show"]),

            cmap="YlGnBu", annot=True,  cbar=True, fmt='4.0f');

bottom, top = ax.get_ylim()

ax.set_title("Age Group vs Medical No Show")

ax.set_ylim(bottom + 0.1, top - 0.1);

ax.get_figure().savefig('./age_group_noshow.png');
#2. Lets create a new feature for number of days in advance a booking was made. If it is zero, it is same day appointment



dt = lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").date()  #"%Y-%m-%dT%H:%M:%SZ"

df['AdvanceDays'] = (df['AppointmentDay'].apply(dt) - df['ScheduledDay'].apply(dt)).dt.days
# Before we deep dive into AdvanceDays booking, lets see how that correlates to No-Show. 

fig, ax = plt.subplots(1, 2,figsize=(8,4))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.75, hspace=None)

# The parameter meanings (and suggested defaults) are:



# left  = 0.125  # the left side of the subplots of the figure

# right = 0.9    # the right side of the subplots of the figure

# bottom = 0.1   # the bottom of the subplots of the figure

# top = 0.9      # the top of the subplots of the figure

# wspace = 0.2   # the amount of width reserved for blank space between subplots

# hspace = 0.2   # the amount of height reserved for white space between subplots



ax = ax.ravel()

ax[0].hist(df[df['No-show']=='No']['AdvanceDays'], bins=75)

ax[1].hist(df[df['No-show']=='Yes']['AdvanceDays'], bins=75);

#bottom, top = ax.get_ylim()

ax[0].set_title("Histogram - Honor Appointments")

ax[1].set_title("Histogram - Medical No Show");



# We see that the number of appointments booked for various advance booking are not same. 

# Also the No-show fluctuates across these days. 

# Hence lets create custom bins for the booking days and then see if we are able to see any pattern

plt.savefig('./adv_booking_hist.png')
#3. Create advance_days_bin for grouping the booked in advance days into logical buckets

def create_advance_days_bins(advance_days):

    """

        This function creates a new feature to the input dataframe based on the AdvanceDays.



        It creates categorical values based on AdvanceDays. It categorizes the Age into 1 of the 10 categories.

        #   90-Days+               -> More than or equal to 90 days in advance

        #   30-Days+               -> More than or equal to 30 days in advance

        #   7-Days+                -> More than or equal to 7 days in advance

        #   6-Days                 -> 6 days in advance

        #   5-Days                 -> 5 days in advance

        #   4-Days                 -> 4 days in advance

        #   3-Days                 -> 3 days in advance

        #   2-Days                 -> 2 days in advance

        #   1-Days                 -> 1 days in advance

        #   0-Days                 -> Same day



        INPUT:

        advance_days - Number of days in advance an appointment is booked



        OUTPUT:

        adv_bin - AdvanceDays Bin



    """

    

    if advance_days >= 90:

        adv_bin = '90-Days+'

    elif advance_days >= 30:

        adv_bin = '30-Days+'             

    elif advance_days >= 7:

        adv_bin = '07-Days+'

    elif advance_days == 6:

        adv_bin = '06-Days'

    elif advance_days == 5:

        adv_bin = '05-Days'

    elif advance_days == 4:

        adv_bin = '04-Days'

    elif advance_days == 3:

        adv_bin = '03-Days'

    elif advance_days == 2:

        adv_bin = '02-Days'

    elif advance_days == 1:

        adv_bin = '01-Day'

    else:

        adv_bin = '00-SameDay'        

    return adv_bin



df['adv_bin'] = df['AdvanceDays'].apply(create_advance_days_bins)
# Analysis 4.3

# Crosstab between AdvanceBin vs No-show normalized to percentages to identify its impact. 

# We then use Seaborn heatmap to visually see the output.



ax = sns.heatmap(pd.crosstab([df.adv_bin], df['No-show'],rownames=["Advance Booking"], colnames=["No Show"]),

            cmap="YlGnBu", annot=True,  cbar=True, fmt='4.0f');

bottom, top = ax.get_ylim()

ax.set_title("Advance Booking vs Medical No Show")

ax.set_ylim(bottom + 0.1, top - 0.1);

ax.get_figure().savefig('./advance_bookins_noshow.png');
def model_fit_predict(model, X_train, y_train, X_test, y_test, target_names, model_name):

    #5. Apply model on train dataset

    model.fit(X_train, y_train)



    #6. Test model using test dataset

    y_pred = model.predict(X_test)



    #7. Measure the score of accuracy

    print('{1} Evaluation...Accuracy Score = {0:3.2f}'.format(accuracy_score(y_test, y_pred), model_name))



    #8. Confusion Matrix of the prediction

    print(confusion_matrix(y_test, y_pred ))



    #9. Classification Report of the prediction

    print(classification_report(y_test, y_pred,target_names=target_names))    

    return y_pred



def logistic_regression(X_train, y_train, X_test, y_test, target_names, model_name):

    # Logistics Regression Model

    reg_lr = LogisticRegression(random_state = 0)    

    y_pred_lr = model_fit_predict(reg_lr, X_train, y_train, X_test, y_test, target_names, model_name)

    return y_pred_lr



def random_forest_classifier(X_train, y_train, X_test, y_test, target_names, model_name):

    # Random Forest Classification Model

    clf_rfc = RandomForestClassifier(random_state = 0, n_estimators=10)

    y_pred_rfc = model_fit_predict(clf_rfc, X_train, y_train, X_test, y_test, target_names, model_name)

    return y_pred_rfc



def naive_bayes_classifier(X_train, y_train, X_test, y_test, target_names, model_name):

    # Naive Bayes Gaussian Model

    #5. Apply model on train dataset

    clf_nb = GaussianNB()

    y_pred_nb = model_fit_predict(clf_nb, X_train, y_train, X_test, y_test, target_names, model_name)

    return y_pred_nb



def run_all_models(X_train, y_train, X_test, y_test, target_names):

    #5. Model Creation and Validation

    y_pred_lr  = logistic_regression(X_train, y_train, X_test, y_test, target_names, 'Logistic Regression')

    y_pred_rfc = random_forest_classifier(X_train, y_train, X_test, y_test, target_names,'Random Forest')

    y_pred_nb  = naive_bayes_classifier(X_train, y_train, X_test, y_test, target_names,'Naive Bayes')

    

# Lets create the Model

#df = df_bkup

#df_bkup = df

#1. Identify all unnecessary columns in the Dataframe that are not required or add any value to the findings. Remove them.

df = df.drop(columns=['PatientId','AppointmentID','ScheduledDay','AppointmentDay','Age','weekday','SMS_Desc','AdvanceDays'])



#2. Identify all categorical columns and convert them using One-Hot encoding with dropFirst = True

df = pd.get_dummies(df, columns = ['Gender','Neighbourhood','Age_Group','adv_bin','No-show'], drop_first=True)



#3. Separate the Dataframe into X input and y output series.

X = df.drop(['No-show_Yes'], axis=1)

y = df['No-show_Yes']

target_names = ['Show', 'No-Show']



#4. Split into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.3)



#5. Model Execution

run_all_models(X_train, y_train, X_test, y_test, target_names)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)





#6. Resample - Undersampling

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.3)



df_train = pd.concat((X_train, pd.DataFrame(y_train.T)),axis=1)

undersample_size = y_train.value_counts().min()

# Display new class counts

print (y_train.value_counts())



df_train_major = df_train[df_train['No-show_Yes']==0]

df_train_minor = df_train[df_train['No-show_Yes']==1]



df_train_major_undersample = resample(df_train_major, 

                                 replace=True,     

                                 n_samples=undersample_size,    

                                 random_state=0) 

df_train = pd.concat([df_train_major_undersample, df_train_minor], axis = 0)

X_train = df_train.drop(columns = ['No-show_Yes'])

y_train = df_train['No-show_Yes']

 

# Display new class counts

print (y_train.value_counts())



#5. Model Execution

run_all_models(X_train, y_train, X_test, y_test, target_names)

#7. Resample - Oversampling

from sklearn.utils import resample

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.3)



df_train = pd.concat((X_train, pd.DataFrame(y_train.T)),axis=1)

oversample_size = y_train.value_counts().max()

# Display new class counts

print (y_train.value_counts())



df_train_major = df_train[df_train['No-show_Yes']==0]

df_train_minor = df_train[df_train['No-show_Yes']==1]



df_train_minor_oversample = resample(df_train_minor, 

                                 replace=True,     

                                 n_samples=oversample_size,    

                                 random_state=0) 

df_train = pd.concat([df_train_major, df_train_minor_oversample], axis = 0)

X_train = df_train.drop(columns = ['No-show_Yes'])

y_train = df_train['No-show_Yes']

 

# Display new class counts

print (y_train.value_counts())



#5. Model Execution

run_all_models(X_train, y_train, X_test, y_test, target_names)
#8. Resample - Synthetic sampling

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.3)



#SMOTE

import sys

#!{sys.executable} -m pip install imblearn

from imblearn.over_sampling import SMOTE



sm = SMOTE(random_state=12)

# Display new class counts

print (y_train.value_counts())



X_train, y_train = sm.fit_sample(X_train, y_train)

# Display new class counts

print (np.bincount(y_train))



#print (Y_train.value_counts() , np.bincount(y_train_res))



#5. Model Execution

run_all_models(X_train, y_train, X_test, y_test, target_names)