import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv('../input/pulsar_stars.csv')
df.head()
df.shape
df.describe()
df.isnull().sum()
cols = ['mean_profile', 'std_profile', 'Excess_kurtosis_profile', 'skew_profile', 'mean_dmsnr', 'std_dmsnr', 'excess_kurtosis_dmsnr', 'skew_dmsnr', 'target_class']
print(len(cols))
data = np.array(df)
df = pd.DataFrame(data, columns = cols)
df.head()
df.shape
targets = np.array(df["target_class"])
sns.distplot(targets, kde = False, color = "r", axlabel = "Target Class").set_title("Pulsar (1) vs Non-Pulsar (0) Count")

star_data = []
notstar_data =[]

for i in range(df["target_class"].count()):
    if df["target_class"][i] == 1:
        star_data.append(df.loc[i])
    else:
        notstar_data.append(df.loc[i])
    
len(star_data) + len(notstar_data)
dfstar = pd.DataFrame(star_data, columns = cols )
dfnotstar = pd.DataFrame(notstar_data, columns = cols)
dfstar.head()
dfstar = dfstar.reset_index(drop = True)
dfnotstar = dfnotstar.reset_index(drop = True)
dfstar.head()
dfnotstar.head()
dfnotstar.describe()
dfstar.describe()
mean_star_profile = np.array(dfstar["mean_profile"])
mean_notstar_profile = np.array(dfnotstar["mean_profile"])
sns.distplot(mean_star_profile, color = "b", label = "Pulsar", axlabel = "Mean").set_title("Means of Integrated Profiles")
sns.distplot(mean_notstar_profile, color = "r", label = "Not Pulsar")
plt.legend()
std_star_profile = np.array(dfstar["std_profile"])
std_notstar_profile = np.array(dfnotstar["std_profile"])
sns.distplot(std_star_profile, color = "b", label = "Pulsar", axlabel = "Standard Deviation").set_title("Standard Deviations of Integrated Profiles")
sns.distplot(std_notstar_profile, color = "r", label = "Not Pulsar")
plt.legend()
exkurt_star_profile = np.array(dfstar["Excess_kurtosis_profile"])
exkurt_notstar_profile = np.array(dfnotstar["Excess_kurtosis_profile"])
sns.distplot(exkurt_star_profile, color = "b", axlabel = "Excess Kurtosis", label = "Pulsar").set_title("Excess Kurtosis for Integrated Profiles")
sns.distplot(exkurt_notstar_profile, color = "r", label = "Not Pulsar")
plt.legend()
skew_star_profile = np.array(dfstar["skew_profile"])
skew_notstar_profile = np.array(dfnotstar["skew_profile"])
sns.distplot(skew_star_profile, color = "b", axlabel = "Skewness", label = "Pulsar").set_title("Skewness for Integrated Profiles")
sns.distplot(skew_notstar_profile, color = "r", label = "Not Pulsar")
plt.legend()
mean_star_dmsnr = np.array(dfstar["mean_dmsnr"])
mean_notstar_dmsnr = np.array(dfnotstar["mean_dmsnr"])
sns.distplot(mean_star_dmsnr, color = "b", axlabel = "Mean", label = "Pulsar").set_title("Means for DM-SNR Curves")
sns.distplot(mean_notstar_dmsnr, color = "r", label = "Not Pulsar")
plt.legend()
std_star_dmsnr = np.array(dfstar["std_dmsnr"])
std_notstar_dmsnr = np.array(dfnotstar["std_dmsnr"])
sns.distplot(std_star_dmsnr, color = "b", axlabel = "Standard Deviation", label = "Pulsar").set_title("Standard Deviations for DM-SNR Curves")
sns.distplot(std_notstar_dmsnr, color = "r", label = "Not Pulsar")
plt.legend()
exkurt_star_dmsnr = np.array(dfstar["excess_kurtosis_dmsnr"])
exkurt_notstar_dmsnr = np.array(dfnotstar["excess_kurtosis_dmsnr"])
sns.distplot(exkurt_star_dmsnr, color = "b", axlabel = "Excess Kurtosis", label = "Pulsar").set_title("Excess Kurtosis for DM-SNR Curves")
sns.distplot(exkurt_notstar_dmsnr, color = "r", label = "Not Pulsar")
plt.legend()
skew_star_dmsnr = np.array(dfstar["skew_dmsnr"])
skew_notstar_dmsnr = np.array(dfnotstar["skew_dmsnr"])
sns.distplot(skew_star_dmsnr, color = "b", axlabel = "Skewness", label = "Pulsar").set_title("Skewness for DM-SNR Curves")
sns.distplot(skew_notstar_dmsnr, color = "r", label = "Not Pulsar")
plt.legend()
df.head()
sns.heatmap(df[["mean_profile", "std_profile", "Excess_kurtosis_profile", "skew_profile", "mean_dmsnr", "std_dmsnr", "excess_kurtosis_dmsnr", "skew_dmsnr", "target_class"]].corr(), annot = True)
sns.set(style = "dark")
from sklearn import svm
from sklearn.model_selection import train_test_split

X = np.array(df.drop("target_class", axis = 1))
y = np.array(df["target_class"])
X_train, X_test, y_train, y_true = train_test_split(X, y, test_size = 0.27)
clf = svm.SVC(kernel = 'linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier()
dct.fit(X_train, y_train)
y_pred1 = dct.predict(X_test)
accuracy_score(y_true, y_pred1)
print(classification_report(y_true, y_pred1))
