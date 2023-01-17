# Veri temizleme için gerekli kütüphaneler import edildi.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Kullanılacak scikit-learn kütüphaneleri import edildi.
# Kullanıldıkları yerlerde de import edildi.
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
# Gerekli veri seti alındı.
df = pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv");
pd.options.display.max_columns = None
df.head()
# df.info() tam olarak istenilen tabloyu vermediği için yeni bir tablo oluşturuldu.
col_info = pd.concat([df.count().rename("count"), df.nunique().rename("unique count"), df.isna().sum().rename("null count"), df.dtypes.rename("types")], axis=1)
pd.options.display.max_rows = None
col_info
# Respondent tahminleme için anlamsız bir sütun olduğu için silindi.
# Diğerleri çok fazla unique ve/veya null değeri olduğu için silindi.
df = df.drop(["Respondent", "EduOther", "DevType", "LastInt",
              "JobFactors", "ResumeUpdate", "CurrencySymbol", "CurrencyDesc",
              "CompTotal", "CompFreq", "WorkChallenge", "LanguageWorkedWith",
              "LanguageDesireNextYear", "DatabaseWorkedWith",
              "DatabaseDesireNextYear", "PlatformWorkedWith",
              "PlatformDesireNextYear", "WebFrameWorkedWith",
              "WebFrameDesireNextYear", "MiscTechWorkedWith",
              "MiscTechDesireNextYear", "DevEnviron", "Containers",
              "SocialMedia", "SOVisitTo", "SONewContent", "Ethnicity"], axis=1)

# YearsCode, Age1stCode, YearsCodePro, SOVisit1st genellikle sayısal değerlerden oluşuyor.
# Sayısal değer olmayanlar, mantıklı sayılar değerlere çevrilerek sütun, tamamen sayısal değerlere çevrildi.
df["YearsCode"] = df["YearsCode"].apply(lambda x: 0 if x == "Less than 1 year" else x)
df["YearsCode"] = df["YearsCode"].apply(lambda x: 51 if x == "More than 50 years" else x)
df["YearsCode"] = pd.to_numeric(df["YearsCode"])

df["Age1stCode"] = df["Age1stCode"].apply(lambda x: 4 if x == "Younger than 5 years" else x)
df["Age1stCode"] = df["Age1stCode"].apply(lambda x: 86 if x == "Older than 85" else x)
df["Age1stCode"] = pd.to_numeric(df["Age1stCode"])

df["YearsCodePro"] = df["YearsCodePro"].apply(lambda x: 0 if x == "Less than 1 year" else x)
df["YearsCodePro"] = df["YearsCodePro"].apply(lambda x: 51 if x == "More than 50 years" else x)
df["YearsCodePro"] = pd.to_numeric(df["YearsCodePro"])

df["SOVisit1st"] = df["SOVisit1st"].apply(lambda x: -1 if x == "I don't remember" else x)
df["SOVisit1st"] = pd.to_numeric(df["SOVisit1st"])

# CareerSat sorularda kullanılabilecek şekilde değiştirildi.
df["CareerSat"] = df["CareerSat"].apply(lambda x: 5 if x in ["Very satisfied"] else x)
df["CareerSat"] = df["CareerSat"].apply(lambda x: 4 if x in ["Slightly satisfied"] else x)
df["CareerSat"] = df["CareerSat"].apply(lambda x: 3 if x in ["Neither satisfied nor dissatisfied"] else x)
df["CareerSat"] = df["CareerSat"].apply(lambda x: 2 if x in ["Slightly dissatisfied"] else x)
df["CareerSat"] = df["CareerSat"].apply(lambda x: 1 if x in ["Very dissatisfied"] else x)
df["CareerSat"] = pd.to_numeric(df["CareerSat"])

# ConvertedComp, WorkWeekHrs, CodeRevHrs, Age zaten sayısal sütunlar olduğu için işlem yapılmadı.

# OpenSourcer sorularda kullanıldığı için sonra işlem yapılacak.
# Country sütunu da tahminleme için anlamsız ancak sorularda kullanıldığı için sonra işlem yapılacak.

df.head()
# NA değerler sayısal sütunlarda, sütunların ortalaması ile,
# kategorik (metinsel) sütunlarda, sütunlarda en çok tekrar eden değer ile dolduruldu.
cateogrical_columns = df.select_dtypes(include=["object"]).columns.tolist()
numeric_columns = df.select_dtypes(include=["float64"]).columns.tolist()

for column in df:
    if df[column].isnull().any():
        if column in cateogrical_columns:
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].mean())

df.head()
# Sonuç olarak NA değerler olmayan net bir veri seti elde edildi.
plt.figure(figsize=(20,8))
sns.heatmap(df.isnull())
# Kategorik değerler sayısal olarak ifade edildi.
df = pd.get_dummies(df, columns=["MainBranch", "Hobbyist", "OpenSource", "Employment",
                                 "Student", "EdLevel", "UndergradMajor", "OrgSize",
                                 "JobSat", "MgrIdiot", "MgrMoney", "MgrWant",
                                 "JobSeek", "LastHireDate", "FizzBuzz", "WorkPlan",
                                 "WorkRemote", "WorkLoc", "ImpSyn", "CodeRev", "UnitTests",
                                 "PurchaseHow", "PurchaseWhat", "OpSys", "BlockchainOrg",
                                 "BlockchainIs", "BetterLife", "ITperson", "OffOn", "Extraversion",
                                 "ScreenName", "SOVisitFreq", "SOFindAnswer", "SOTimeSaved", 
                                 "SOHowMuchTime", "SOAccount", "SOPartFreq", "SOJobs", "EntTeams",
                                 "SOComm", "WelcomeChange", "Gender", "Trans", "Sexuality",
                                 "Dependents", "SurveyLength", "SurveyEase"], drop_first=True)
df.head()
# Temiz veri setinin boyutları
df.shape
# Country sütunu bu soru için gereksiz olduğu için silindi.
soru1 = df.drop(["Country"], axis=1)
# OpenSourcer sütunu sorudaki istediği gibi değiştirildi.
soru1["OpenSourcer"] = soru1["OpenSourcer"].apply(lambda x: "Katkıda bulunmuyor" if x in ["Never", "Less than once per year"] else x)
soru1["OpenSourcer"] = soru1["OpenSourcer"].apply(lambda x: "Katkıda bulunuyor" if x in ["Less than once a month but more than once per year", "Once a month or more often"] else x)
soru1.head()
# Tahminlemeye uygun hale gelmesi için OpenSourcer sütunu sayısal değerlere çevrildi.
soru1["OpenSourcerPred"] = soru1["OpenSourcer"].apply(lambda x: 0 if x == "Katkıda bulunmuyor" else 1)
soru1.head()
# Veri seti, X (tahminleme için kullanılacak özellikler) ve y (tahminlenecek özellik) veri setleri olmak üzere 2 veri setine ayrıldı.
X = soru1.drop(["OpenSourcer", "OpenSourcerPred"], axis=1)
y = soru1["OpenSourcerPred"]
# Veri seti, train ve test veri setleri olmak üzere 2 veri setine ayrıldı.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# Tahminleme için gerekli kütüphaneler import edildi.
# Tahminlemenin önceki denemelerdeki başarısızlığından dolayı ilk önce scale işlemine tabii tutalan veri seti,
# Ardından model kullanılarak eğitildi.
# Bu işlemi otomatikleştirmek içine pipeline kullanıldı.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
svc = Pipeline([
    ("scale", StandardScaler()),
    ("model", LinearSVC(max_iter=100000, verbose=True))
])
# Train veri setileri ile eğitildi.
svc.fit(X_train, y_train)
# Eğitilen veri seti tahminlemesi yapıldı.
predictions = svc.predict(X_test)
# Tahminlemenin doğruluğu gözlemlendi.
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error, mean_squared_error
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print("Accuracy: ", accuracy_score(y_test, predictions))
print("Mean Absolute Error: ", mean_absolute_error(y_test, predictions))
print("Mean Squared Error: ", mean_squared_error(y_test, predictions))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, predictions)))
df.head()
# OpenSourcer, 1. soruda kullanılması için bırakılmıştı, ancak bu soruda sayısal değerlere çevrilerek silindi.
soru2 = pd.get_dummies(df, columns=["OpenSourcer"], drop_first=True)
# Country sütunu bu soru için gereksiz olduğu için silindi.
soru2 = soru2.drop(["Country"], axis=1)
# CareerSat sütunu tahminlenecek, yukarıda bu sütundaki NA içeren hücreler sütunun ortalaması ile doldurulmuştu.
# Şu an ortalamalar daha iyi sınıflandırmak için 3'e çevriliyor.
soru2["CareerSat"] = soru2["CareerSat"].apply(lambda x: 3 if x > 3 and x < 4 else x)
soru2.head()
# Veri seti, X (tahminleme için kullanılacak özellikler) ve y (tahminlenecek özellik) veri setleri olmak üzere 2 veri setine ayrıldı.
X = soru2.drop(["CareerSat"], axis=1)
y = soru2["CareerSat"]
# Veri seti, train ve test veri setleri olmak üzere 2 veri setine ayrıldı.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# Tahminleme için gerekli kütüphaneler import edildi.
# Tahminlemenin önceki denemelerdeki başarısızlığından dolayı ilk önce scale işlemine tabii tutalan veri seti,
# Ardından model kullanılarak eğitildi.
# Bu işlemi otomatikleştirmek içine pipeline kullanıldı.
from sklearn.linear_model import LogisticRegression
logr = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=99999999)),
])
# Train veri setileri ile eğitildi.
logr.fit(X_train, y_train)
# Eğitilen veri seti tahminlemesi yapıldı.
predictions = logr.predict(X_test)
# Tahminlemenin doğruluğu gözlemlendi.
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error, mean_squared_error
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
# Tahminlemenin doğruluğu gözlemlendi.
print("Accuracy: ", accuracy_score(y_test, predictions))
print("Mean Absolute Error: ", mean_absolute_error(y_test, predictions))
print("Mean Squared Error: ", mean_squared_error(y_test, predictions))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, predictions)))
df.head()
# OpenSourcer, 1. soruda kullanılması için bırakılmıştı, ancak bu soruda sayısal değerlere çevrilerek silindi.
soru3 = pd.get_dummies(df, columns=["OpenSourcer"], drop_first=True)
# Country sütunu bu soru için gereksiz olduğu için silindi.
soru3 = soru3.drop(["Country"], axis=1)
soru3.head()
# Veri seti, X (tahminleme için kullanılacak özellikler) ve y (tahminlenecek özellik) veri setleri olmak üzere 2 veri setine ayrıldı.
X = soru3.drop(["ConvertedComp"], axis=1)
y = soru3["ConvertedComp"]
# Veri seti, train ve test veri setleri olmak üzere 2 veri setine ayrıldı.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# Tahminleme için gerekli kütüphaneler import edildi.
# Tahminlemenin önceki denemelerdeki başarısızlığından dolayı ilk önce scale işlemine tabii tutalan veri seti,
# Ardından model kullanılarak eğitildi.
# Bu işlemi otomatikleştirmek içine pipeline kullanıldı.
from sklearn.linear_model import LinearRegression
linr = Pipeline([
    ("scale", StandardScaler()),
    ("model", LinearRegression())
])
# Train veri setileri ile eğitildi.
linr.fit(X_train, y_train)
# Eğitilen veri seti tahminlemesi yapıldı.
predictions = linr.predict(X_test)
# Tahminlemenin doğruluğu gözlemlendi.
print("Mean Absolute Error: ", mean_absolute_error(y_test, predictions))
print("Mean Squared Error: ", mean_squared_error(y_test, predictions))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, predictions)))
# Tahminlemenin doğruluğu görsel olarak gözlemlendi.
sns.scatterplot(y_test, predictions)
df.head()
# OpenSourcer, 1. soruda kullanılması için bırakılmıştı, ancak bu soruda sayısal değerlere çevrilerek silindi.
soru4 = pd.get_dummies(df, columns=["OpenSourcer"], drop_first=True)
soru4.head()
# Country sütunu bu soruda kullanılacak olan sütun, o sütun sayesinde sadece Türkiye'deki insanlar seçildi.
soru4 = soru4[soru4["Country"] == "Turkey"]
soru4.head()
# Daha sonra sütun sayısal değerli bir sütun olmadığı için silindi.
soru4 = soru4.drop(["Country"], axis=1)
soru4.head()
# ConvertedComp sütunu sorudaki istediği gibi değiştirildi.
soru4["ConvertedComp"] = soru4["ConvertedComp"].apply(lambda x: "Yüksek" if x >= 18000 else "Düşük")
soru4.head()
# Tahminlemeye uygun hale gelmesi için ConvertedComp sütunu sayısal değerlere çevrildi.
soru4["ConvertedCompPred"] = soru4["ConvertedComp"].apply(lambda x: 0 if x == "Düşük" else 1)
soru4.head()
# Yüksek sayısı, Düşük sayısının nerdeyse 3 katı olduğu için tahminleme Yüksek olmaya her zaman daha yatkın olacaktır.
soru4["ConvertedComp"].value_counts()
# Veri seti, X (tahminleme için kullanılacak özellikler) ve y (tahminlenecek özellik) veri setleri olmak üzere 2 veri setine ayrıldı.
X = soru4.drop(["ConvertedComp", "ConvertedCompPred"], axis=1)
y = soru4["ConvertedCompPred"]
# Veri seti, train ve test veri setleri olmak üzere 2 veri setine ayrıldı.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# Tahminleme için gerekli kütüphaneler import edildi.
# Tahminlemenin önceki denemelerdeki başarısızlığından dolayı ilk önce scale işlemine tabii tutalan veri seti,
# Ardından model kullanılarak eğitildi.
# Bu işlemi otomatikleştirmek içine pipeline kullanıldı.
svc = Pipeline([
    ("scale", StandardScaler()),
    ("model", LinearSVC(max_iter=100000, verbose=True))
])
# Train veri setileri ile eğitildi.
svc.fit(X_train, y_train)
# Eğitilen veri seti tahminlemesi yapıldı.
predictions = svc.predict(X_test)
# Tahminlemenin doğruluğu gözlemlendi.
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print("Accuracy: ", accuracy_score(y_test, predictions))
print("Mean Absolute Error: ", mean_absolute_error(y_test, predictions))
print("Mean Squared Error: ", mean_squared_error(y_test, predictions))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, predictions)))
# GridsearchCV ile SVC
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
param_grid = {"C":[0.1,1,10,100,1000], "gamma":[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=3)
grid.cv = 3
grid.fit(X_train, y_train)
grid.best_params_
grid.best_estimator_
grid_pred = grid.predict(X_test)
print(confusion_matrix(y_test, grid_pred))
print(classification_report(y_test, grid_pred))
print("Accuracy: ", accuracy_score(y_test, grid_pred))
print("Mean Absolute Error: ", mean_absolute_error(y_test, grid_pred))
print("Mean Squared Error: ", mean_squared_error(y_test, grid_pred))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, grid_pred)))
df.head()
# OpenSourcer, 1. soruda kullanılması için bırakılmıştı, ancak bu soruda sayısal değerlere çevrilerek silindi.
soru5 = pd.get_dummies(df, columns=["OpenSourcer"], drop_first=True)
# Country sütunu bu soru için gereksiz olduğu için silindi.
soru5 = soru5.drop(["Country"], axis=1)
soru5.head()
# Age sütunu kullanılarak BirthDate sütunu oluşturuldu.
soru5["BirthDate"] = soru5["Age"].apply(lambda x: 2020 - x)
# Daha sonra Age sütunu silindi.
soru5 = soru5.drop(["Age"], axis=1)
soru5.head()
# Veri seti, X (tahminleme için kullanılacak özellikler) ve y (tahminlenecek özellik) veri setleri olmak üzere 2 veri setine ayrıldı.
X = soru5.drop(["BirthDate"], axis=1)
y = soru5["BirthDate"]
# Veri seti, train ve test veri setleri olmak üzere 2 veri setine ayrıldı.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# Tahminleme için gerekli kütüphaneler import edildi.
# Tahminlemenin önceki denemelerdeki başarısızlığından dolayı ilk önce scale işlemine tabii tutalan veri seti,
# Ardından model kullanılarak eğitildi.
# Bu işlemi otomatikleştirmek içine pipeline kullanıldı.
linr = Pipeline([
    ("scale", StandardScaler()),
    ("model", LinearRegression())
])
# Train veri setileri ile eğitildi.
linr.fit(X_train, y_train)
# Eğitilen veri seti tahminlemesi yapıldı.
predictions = linr.predict(X_test)
# Tahminlemenin doğruluğu gözlemlendi.
print("Mean Absolute Error: ", mean_absolute_error(y_test, predictions))
print("Mean Squared Error: ", mean_squared_error(y_test, predictions))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, predictions)))
# Tahminlemenin doğruluğu görsel olarak gözlemlendi.
# Güzel bir dağılım denebilir.
sns.scatterplot(y_test, predictions)