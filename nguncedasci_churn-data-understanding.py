import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
churn = pd.read_csv("../input/churn-predictions-personal/Churn_Predictions.csv")
df=churn.copy()
df.head(3)
df.describe().T
df.info()
# En belirgin olarak,
# Balance ile Number of Product arasında negatif yönlü 0.3, 
# Germany ve Balance arasında ise pozitif yönlü 0.4 lük korelasyon var.
# 0.5'ten yüksek olmadığı için bu korelasyonları dikkate almadım.

plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
# y bağımlı değişkeni("Exited") inceledim.
df["Exited"].value_counts()
# Veri setinde 0-1 dengesi: %20-80
print(2037/9999)
sns.countplot(df['Exited'],label="Count");
# Dummy değişkenleri oluşturdum, satır sayısını gösteren row number değişkenini attım.
df=pd.get_dummies(df,columns=["Geography","Gender"], drop_first=True)
df=df.drop("RowNumber", axis=1)
df=df.reset_index(drop=True)
df.head(2)
df["CreditScore"].describe().T
sns.boxplot(df["CreditScore"])
# Kredi skoru 400'ün altında olanların hepsi churn olmuş
df[df["CreditScore"]<405].apply({"Exited": "value_counts"})
import seaborn as sns
sns.distplot(df["CreditScore"], kde=False)
df["Age"].describe()
# Outlier lar var gibi görünüyor ancak örneğin 90 yaşında bir müşteri olabilir, bunları atmak istemedim.
sns.boxplot(df["Age"]);
sns.distplot(df["Age"], kde=False)
df["Tenure"].value_counts()
sns.countplot(df['Tenure'],label="Count");
df["Balance"].describe()
sns.boxplot(df["Balance"])
# Balance durumlarına göre Exit olma ortalamalarına baktım.
print(df[df["Balance"]>185967]["Exited"].mean())
print(df[df["Balance"]>76485]["Exited"].mean())
print(df[df["Balance"]>97198]["Exited"].mean())
print(df[df["Balance"]>100000]["Exited"].mean())
print(df[df["Balance"]>200000]["Exited"].mean())
sns.distplot(df["Balance"], kde=False)
print(df[df["Balance"]==0].shape)
print(df[df["Balance"]<125000].shape)
print(df[df["Balance"]<250000].shape)
# Balance değerleri 0 olmayanların, 0 olanlara oranla churn olma olasılıkları daha yüksek.
# Preprocessing'te bununla ilgili bir değişken oluşturdum.
print(df[df["Balance"]==0]["Exited"].mean())
print(df[df["Balance"]!=0]["Exited"].mean())
df["NumOfProducts"].value_counts()
df.groupby("NumOfProducts").agg({"Exited":"count"})
# Number of Products 4'se'kesin churn, 3'se %82 olasılıkla churn, 2 ise yüksek ihtimal churn değil, 1 ise %27 ihtimal churn.
# Preprocessing'te bu sıralamaya göre yeni bir değişken oluşturdum. (NOP* isminde)
print(df[(df["NumOfProducts"]==1)]["Exited"].mean())
print(df[(df["NumOfProducts"]==2)]["Exited"].mean())
print(df[(df["NumOfProducts"]==3)]["Exited"].mean())
print(df[(df["NumOfProducts"]==4)]["Exited"].mean())

#^Müşterilerin %70'inin kredi kartı var. 
print(df["HasCrCard"].value_counts())
print("0-1 proportion:",7055/9999)
sns.countplot(df['HasCrCard'],label="Count");
# Kredi kartı olup olmaması ile churn olması arasında ilişki vardır diye düşündüm ama yokmuş.
df['HasCrCard'].corr(df['Exited'])
# Almanların terk olasılığı, diğer milletlere göre daha yüksek
# Almanlarda %32 terk- %68 kalıyor
print(df[df["Geography_Germany"]==1].shape)
print(df[(df["Geography_Germany"]==1) & (df["Exited"]==1)].shape)
print(df[(df["Geography_Germany"]==1) & (df["Exited"]==0)].shape)
print(1695/2509)
print(814/2509)
# İspanyollarda %17 terk- %83 kalıyor
print(df[df["Geography_Spain"]==1].shape)
print(df[(df["Geography_Spain"]==1) & (df["Exited"]==1)].shape)
print(df[(df["Geography_Spain"]==1) & (df["Exited"]==0)].shape)
print(413/2477)
print(2064/2477)
# Fransızlarda %16 terk- %83 kalıyor
print(df[(df["Geography_Spain"]==0) & (df["Geography_Germany"]==0)].shape)
print(df[(df["Geography_Spain"]==0) & (df["Geography_Germany"]==0)& (df["Exited"]==1)].shape)
print(df[(df["Geography_Spain"]==0) & (df["Geography_Germany"]==0)& (df["Exited"]==0)].shape)
print(810/5014)
print(4204/5014)
#Müşterilerin %55'i erkek, %45'i kadın
print(df["Gender_Male"].value_counts())
print(5457/9999)
sns.countplot(df["Gender_Male"],label="Count");
