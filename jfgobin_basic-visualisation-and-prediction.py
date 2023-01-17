# Import some libraries

import pandas as pd

import numpy as np

import seaborn as sbs

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier



# Read the dataset

dts = pd.read_csv("../input/No-show-Issue-Comma-300k.csv")

# Some early processing, such as binning

# Create categories

# Binning for Awaiting Time

# We consider Immediate for within 2 days, then within the week, within two weeks, within

# the month, within the trimester, within half year and everything above

bins = [-99999, -180, -90, -30, -14, -7, -2, 0]

labels = ["More than half year", "Half year", "Trimester", 

          "Month", "TwoWeeks", "Week", "Immediate"]

wait_period = pd.cut(dts.AwaitingTime, bins, labels=labels)

dts['Wait_period'] = wait_period

# Binning for age, based loosely on typical categories

# Parents tend to be more concerned for babies, Infants can't really tell what's wrong

# Child do but need parents to go to an appointment

# Teenagers are suffering from other ailment (beginning of puberty and such, memories ...)

# Young Adults may be in studies and have to work

bins = [0,2,6,12,18,40,65,np.inf]

labels=["Baby","Infant","Child","Teenager","Young adults", "Adult","Elder"]

age_cat = pd.cut(dts.Age,bins,labels=labels)

dts['Age_cat'] = age_cat

# Create a boolean for Status, with True if the patient showed up

dts.eval("Status_B = Status == 'Show-Up'", inplace=True)

# Extract the month of the visit

dts['Month'] = dts['ApointmentData'].apply(lambda x: x[5:7])

#Information about when the registration was made

dts['Reg_month'] = dts['AppointmentRegistration'].apply(lambda x: x[5:7])

dts['Reg_hour'] = dts['AppointmentRegistration'].apply(lambda x: x[11:13])

impact = {}

# How does the Status distribute?

groups = dts.groupby(['Status'])

gps = groups.size()

ax = sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()
# How does the Wait_Period distribute?

groups = dts.groupby(['Wait_period'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['Wait_period', 'Status','Smokes']].groupby(['Wait_period', 'Status']).count()

groups = groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

impact['Wait_period'] = std

sbs.barplot(y="Smokes", x="Wait_period", hue="Status", data=groups)

sbs.plt.show()
# How does the Day of the week distribute?

groups = dts.groupby(['DayOfTheWeek'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['DayOfTheWeek', 'Status','Smokes']].groupby(['DayOfTheWeek', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

impact['DayOfTheWeek'] = std

sbs.barplot(y="Smokes", x="DayOfTheWeek", hue="Status", data=groups)

sbs.plt.show()
# How does the appointment month distribute?

groups = dts.groupby(['Month'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['Month', 'Status','Smokes']].groupby(['Month', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

impact['Month'] = std

sbs.barplot(y="Smokes", x="Month", hue="Status", data=groups)

sbs.plt.show()
# How does the registration month distribute?

groups = dts.groupby(['Reg_month'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['Reg_month', 'Status','Smokes']].groupby(['Reg_month', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

impact['Reg_month'] = std

sbs.barplot(y="Smokes", x="Reg_month", hue="Status", data=groups)

sbs.plt.show()
# How does the registration hour distribute?

groups = dts.groupby(['Reg_hour'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['Reg_hour', 'Status','Smokes']].groupby(['Reg_hour', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

impact['Reg_hour'] = std

sbs.barplot(y="Smokes", x="Reg_hour", hue="Status", data=groups)

sbs.plt.show()
# How does the Age_category distribute?

groups = dts.groupby(['Age_cat'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['Age_cat', 'Status','Smokes']].groupby(['Age_cat', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

impact['Age_cat'] = std

sbs.barplot(y="Smokes", x="Age_cat", hue="Status", data=groups)

sbs.plt.show()
# How does the Gender distribute?

groups = dts.groupby(['Gender'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['Gender', 'Status','Smokes']].groupby(['Gender', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

impact['Gender'] = std

sbs.barplot(y="Smokes", x="Gender", hue="Status", data=groups)

sbs.plt.show()
# How does the Day of the week distribute?

groups = dts.groupby(['Sms_Reminder'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['Sms_Reminder', 'Status','Smokes']].groupby(['Sms_Reminder', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

impact['Sms_reminder'] = std

sbs.barplot(y="Smokes", x="Sms_Reminder", hue="Status", data=groups)

sbs.plt.show()
# How does the Diabetes distribute?

groups = dts.groupby(['Diabetes'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['Diabetes', 'Status','Smokes']].groupby(['Diabetes', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

impact['Diabetes'] = std

sbs.barplot(y="Smokes", x="Diabetes", hue="Status", data=groups)

sbs.plt.show()
# How does the HiperTension distribute?

groups = dts.groupby(['HiperTension'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['HiperTension', 'Status','Smokes']].groupby(['HiperTension', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

impact['HiperTension'] = std

sbs.barplot(y="Smokes", x="HiperTension", hue="Status", data=groups)

sbs.plt.show()
# How does the Tuberculosis distribute?

groups = dts.groupby(['Tuberculosis'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['Tuberculosis', 'Status','Smokes']].groupby(['Tuberculosis', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

impact['Tuberculosis'] = std

sbs.barplot(y="Smokes", x="Tuberculosis", hue="Status", data=groups)

sbs.plt.show()
# How does the Handicap distribute?

groups = dts.groupby(['Handcap'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['Handcap', 'Status','Smokes']].groupby(['Handcap', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

impact['Handcap'] = std

sbs.barplot(y="Smokes", x="Handcap", hue="Status", data=groups)

sbs.plt.show()
# How does the alcoholism distribute?

groups = dts.groupby(['Alcoolism'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['Alcoolism', 'Status','Smokes']].groupby(['Alcoolism', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

impact['Alcoolism'] = std

sbs.barplot(y="Smokes", x="Alcoolism", hue="Status", data=groups)

sbs.plt.show()
# How does the smoking distribute?

groups = dts.groupby(['Smokes'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['Tuberculosis', 'Status','Smokes']].groupby(['Smokes', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Tuberculosis'].std()

impact['Smokes'] = std

sbs.barplot(y="Tuberculosis", x="Smokes", hue="Status", data=groups)

sbs.plt.show()
# How does the smoking distribute?

groups = dts.groupby(['Scholarship'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['Scholarship', 'Status','Smokes']].groupby(['Scholarship', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

impact['Scholarship'] = std

sbs.barplot(y="Smokes", x="Scholarship", hue="Status", data=groups)

sbs.plt.show()
groups = dts[['Scholarship', 'Smokes', 'Alcoolism', 'Status_B']].groupby(['Scholarship', 'Smokes', 'Alcoolism'])

gps = pd.DataFrame(groups.mean())

gps["counts"] = groups.count()["Status_B"]

gps["Show"] = groups.sum()["Status_B"]

gps
groups = dts.groupby(['Diabetes','HiperTension','Tuberculosis'])

gps = pd.DataFrame(groups.mean())

gps["counts"] = groups.count()["Status_B"]

gps["Show"] = groups.sum()["Status_B"]

gps[['Status_B','counts','Show']]
def class_day(row):

    if row in ("Friday","Monday"):

        return "NearWeekend"

    if row in ("Sunday","Saturday"): #Sunday is way lower than Tuesday but that can be due to limited data

        return "Weekend"

    return "MidWeek"

dts['Day_type'] = dts['DayOfTheWeek'].apply(class_day)



# How does this distribute?

groups = dts.groupby(['Day_type'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['Day_type', 'Status','Smokes']].groupby(['Day_type', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

sbs.barplot(y="Smokes", x="Day_type", hue="Status", data=groups)

sbs.plt.show()

    
def class_hc(row):

    return row>0



dts['HC'] = dts['Handcap'].apply(class_hc)



# How does this distribute?

groups = dts.groupby(['HC'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['HC', 'Status','Smokes']].groupby(['HC', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

sbs.barplot(y="Smokes", x="HC", hue="Status", data=groups)

sbs.plt.show()
def class_se(row):

    K=0

    if row['Scholarship'] == 1:

        K += 4

    if row['Smokes'] == 1:

        K += 2

    if row['Alcoolism'] == 1:

        K += 1

    if K in (0,2):

        return "High"

    if K in (4,1):

        return "Medium"

    return "Low"

dts['Socio_Economics'] = dts[["Scholarship","Smokes","Alcoolism"]].apply(class_se, axis=1)



# How does this distribute?

groups = dts.groupby(['Socio_Economics'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['Socio_Economics', 'Status','Smokes']].groupby(['Socio_Economics', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

sbs.barplot(y="Smokes", x="Socio_Economics", hue="Status", data=groups)

sbs.plt.show()
def class_health(row):

    K=0

    if row['Diabetes'] == 1:

        K += 4

    if row['HiperTension'] == 1:

        K += 2

    if row['Tuberculosis'] == 1:

        K += 1

    if K in (7,5):

        return "High"

    if K in (6,2,4):

        return "Medium"

    return "Low"

dts['Health'] = dts[["Diabetes","HiperTension","Tuberculosis"]].apply(class_health, axis=1)



# How does this distribute?

groups = dts.groupby(['Health'])

gps = groups.size()

sbs.barplot(x=gps.index.tolist(), y=gps.values)

sbs.plt.show()



groups = dts[['Health', 'Status','Smokes']].groupby(['Health', 'Status']).count()

groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()

std = groups.query('Status=="No-Show"')['Smokes'].std()

sbs.barplot(y="Smokes", x="Health", hue="Status", data=groups)

sbs.plt.show()
mldts = dts.copy()





mldts.drop(['AppointmentRegistration','ApointmentData',

            'Status','Diabetes','Alcoolism','HiperTension','Handcap',

            'Smokes','Scholarship','Tuberculosis','Sms_Reminder',

            'Reg_hour'], inplace=True, axis=1)

# Replace gender with a 0/1 variable (0: male, 1: female)

mldts['Gender'] = mldts['Gender'].apply(lambda x: x=="F")

# Convert the categorical columns to dummy encoded ones

DT_age = pd.get_dummies(mldts['Age_cat'], prefix="AC")

#DT_dow = pd.get_dummies(mldts['DayOfTheWeek'], prefix="DOW")

DT_wp = pd.get_dummies(mldts['Wait_period'], prefix="WP")

DT_month = pd.get_dummies(mldts['Month'], prefix="MONTH")

DT_RM = pd.get_dummies(mldts['Reg_month'], prefix="RM")

DT_DT = pd.get_dummies(mldts['Day_type'], prefix="DT")

DT_SE = pd.get_dummies(mldts['Socio_Economics'], prefix="SE")

DT_Health = pd.get_dummies(mldts['Health'], prefix="Health")



mldts = pd.concat([mldts,DT_age, DT_wp, DT_month, DT_RM, DT_DT,

                   DT_SE, DT_Health], axis=1);

mldts.drop(['Age_cat','DayOfTheWeek','Wait_period','Month',

            'Reg_month','Day_type','Socio_Economics','Health'], inplace=True, axis=1)





target = mldts['Status_B']

mldts.drop('Status_B', inplace=True, axis=1)



# Logistic regression



logreg = LogisticRegression()

logreg.fit(mldts, target)

Y_pred = logreg.predict(mldts)

acc_log = logreg.score(mldts, target) * 100

print("Linear regression accuracy: %.2f" % (acc_log))



train_df = mldts.copy()



coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



cfcol = coeff_df.sort_values(by='Correlation', ascending=False)

cfcol
cfcol['abscorr'] = abs(cfcol['Correlation'])

features = cfcol.query('abscorr > 0.15')['Feature'].tolist()

print(cfcol.query('abscorr > 0.15')[['Feature', 'Correlation']])



fmldts=mldts



print("\n\nRandom Forests")

print("==============")

print("max_features=auto")

for nestimators in range(7,12):

    clf = RandomForestClassifier(n_estimators=nestimators)

    scores = cross_val_score(clf, fmldts, target)

    print("Nestimators: %2d, mean score: %.4f"%(nestimators, scores.mean()))

print("max_features=log2")

for nestimators in range(7,12):

    clf = RandomForestClassifier(n_estimators=nestimators, max_features="log2")

    scores = cross_val_score(clf, fmldts, target)

    print("Nestimators: %2d, mean score: %.4f"%(nestimators, scores.mean()))

print("max_feature=None")

for nestimators in range(7,12):

    clf = RandomForestClassifier(n_estimators=nestimators, max_features=None)

    scores = cross_val_score(clf, fmldts, target)

    print("Nestimators: %2d, mean score: %.4f"%(nestimators, scores.mean()))

print("\n\nKNN")

print("===")

for nneigh in range(3,10):

    clf = KNeighborsClassifier(n_neighbors=nneigh)

    scores = cross_val_score(clf, fmldts, target)

    print("N_neighbors: %3d, mean score: %.4f"%(nneigh, scores.mean()))

    
# Let's see if limiting the dataset to age category, socio-economic and current health 

# improves our rating



fmldts = pd.concat([DT_age,DT_SE,DT_Health,DT_month], axis=1)



print("\n\nRandom Forests")

print("==============")

print("max_features=auto")

for nestimators in range(7,12):

    clf = RandomForestClassifier(n_estimators=nestimators)

    scores = cross_val_score(clf, fmldts, target)

    print("Nestimators: %2d, mean score: %.4f"%(nestimators, scores.mean()))

print("max_features=log2")

for nestimators in range(7,12):

    clf = RandomForestClassifier(n_estimators=nestimators, max_features="log2")

    scores = cross_val_score(clf, fmldts, target)

    print("Nestimators: %2d, mean score: %.4f"%(nestimators, scores.mean()))

print("max_feature=None")

for nestimators in range(7,12):

    clf = RandomForestClassifier(n_estimators=nestimators, max_features=None)

    scores = cross_val_score(clf, fmldts, target)

    print("Nestimators: %2d, mean score: %.4f"%(nestimators, scores.mean()))
