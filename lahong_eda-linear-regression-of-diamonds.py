import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pylab as plt



import warnings

warnings.filterwarnings('ignore')
# 데이터 읽기 / 불러오기

diamonds = pd.read_csv("../input/diamonds.csv")

diamonds.head()
# 결측치 확인

diamonds.info()
# 이상치 확인

diamonds.describe()
# 상관관계 파악

plt.figure(figsize=(8, 8))

sns.heatmap(diamonds.corr(), square=True)

plt.show()
sns.pairplot(diamonds)

plt.show()
# price 분포 파악

plt.figure(figsize=(8, 6))

sns.distplot(diamonds["price"])

plt.axvline(np.mean(diamonds["price"]), c='k', ls=":", label="mean")

plt.axvline(np.median(diamonds["price"]), c='k', ls="--", label="median")

plt.legend()

plt.show()
# 중간값을 기준으로 고가/저가 가격으로 분류

high_price = diamonds[diamonds["price"]>2400]

low_price = diamonds[diamonds["price"]<=2400]



plt.figure(figsize=(10, 6))

sns.distplot(low_price["price"])

plt.axvline(np.mean(low_price["price"]), c='b', ls=":", label="low_price_mean")

plt.axvline(np.median(low_price["price"]), c='b', ls="--", label="low_price_median")

plt.legend()

sns.distplot(high_price["price"])

plt.axvline(np.mean(high_price["price"]), c='r', ls=":", label="high_price_mean")

plt.axvline(np.median(high_price["price"]), c='r', ls="--", label="high_price_median")

plt.legend()

plt.show()
# depth, table과 price관계 분석

sns.jointplot("depth", "price", data=diamonds, kind="hex", color="#4CB391")

sns.jointplot("table", "price", data=diamonds, kind="hex", color="#CB2553")

plt.show()
np.corrcoef(diamonds["depth"], diamonds["price"])[1,0],np.corrcoef(diamonds["table"], diamonds["price"])[1,0]
sns.set(style="white")

sns.relplot(x="depth", y="table", hue="color", size="price",

            sizes=(40, 400), alpha=.4, palette="muted", height=5,  data=diamonds)

plt.show()
# x, y, z 의 0값 제거

diamonds[["x","y","z"]]=diamonds[["x","y","z"]].replace(0, np.NaN)

diamonds.dropna(inplace=True)

diamonds.describe()
# x, y, z 관계 파악

plt.scatter("x", "y", data=diamonds)

plt.scatter("y", "z", data=diamonds)

plt.scatter("z", "x", data=diamonds)

plt.show()
# x의 이상치(최댓값) 확인 및 제거

diamonds.sort_values(by=["x"], ascending=False).head(10)
# y의 이상치(최댓값) 확인 및 제거

diamonds.sort_values(by=["y"], ascending=False).head()
diamonds.drop([24067, 49189], inplace=True)
# z의 이상치(최댓값) 확인 및 제거

diamonds.sort_values(by=["z"], ascending=False).head()
diamonds.drop([48410], inplace=True)
# z의 이상치(최솟값) 확인 및 제거

diamonds.sort_values(by=["z"], ascending=True).head()
diamonds.drop([14635, 21654, 20694], inplace=True)
plt.figure()

plt.scatter("x", "y",data=diamonds, alpha=0.3)

plt.scatter("y", "z",data=diamonds, alpha=0.3)

plt.scatter("z", "x",data=diamonds, alpha=0.3)

plt.show()
# x, y, z의상관계수

np.corrcoef(diamonds["x"], diamonds["y"])[1,0], np.corrcoef(diamonds["z"], diamonds["y"])[1,0], np.corrcoef(diamonds["x"], diamonds["z"])[1,0]
# x, y, z의 대체 column 생성

diamonds["volumn"]=diamonds["x"]*diamonds["y"]*diamonds["z"]

diamonds["mean_xyz"]=(diamonds["x"]+diamonds["y"]+diamonds["z"])/3
plt.figure(figsize=(12, 6))

plt.subplot(121)

plt.scatter('volumn', "carat", data=diamonds, color="b", alpha=0.5)

plt.xlabel("volumn")

plt.ylabel("carat")



plt.subplot(122)

plt.scatter( 'mean_xyz', "carat", data=diamonds, color="m", alpha=0.5)

plt.xlabel("mean_xyz")

plt.show()
# volumn, carat 과 price 상관계수

np.corrcoef(diamonds["volumn"], diamonds["carat"])[1,0],np.corrcoef(diamonds["mean_xyz"], diamonds["carat"])[1,0], np.corrcoef(diamonds["carat"], diamonds["price"])[1,0]
plt.figure(figsize=(12, 6))

plt. subplot(121)

sns.lineplot(data = diamonds, x="mean_xyz", y="price")



plt.subplot(122)

sns.lineplot(data = diamonds, x="carat", y="price")



plt.show()
## carat의 크기 1.75를 기준으로 분류

c4= diamonds[["carat", "cut", "color", "clarity", "price"]]

b_carat = c4[c4["carat"]>1.75]

s_carat = c4[c4["carat"]<=1.75]

b_carat.shape, s_carat.shape
colors = [ '#ff6666','#ffcc99', '#66b3ff', '#c2c2f0', '#99ff99']

colors_b = ['#ffcc99', '#ff6666', '#66b3ff', '#c2c2f0', '#99ff99']



plt.subplot(332)

diamonds.cut.value_counts().plot.pie(autopct='%.2f%%', radius=4, colors= colors)

plt.subplot(337)

s_carat.cut.value_counts().plot.pie(autopct='%.1f%%', radius=2.5, colors= colors)

plt.subplot(339)

b_carat.cut.value_counts().plot.pie(autopct='%.1f%%', radius=2.5, colors= colors_b)

plt.show()
sns.barplot(x= "cut", y="price", data = b_carat, palette=("pastel"))

sns.barplot(x= "cut", y="price", data = s_carat)

plt.show()
colors = [ '#c2c2f0',  '#fec8d8', '#f9f9aa', '#66b3ff', '#ffcc99', '#776b3f', '#99ff99', '#ff6666' ]

colors_a = ['#f9f9aa', '#c2c2f0', '#fec8d8',  '#66b3ff', '#ff6666', '#ffcc99',  '#99ff99', '#776b3f' ]



plt.subplot(332)

diamonds.clarity.value_counts().plot.pie(autopct='%.2f%%', radius=2.5, colors=colors)

plt.subplot(337)

s_carat.clarity.value_counts().plot.pie(autopct='%.1f%%', radius=4, colors=colors )

plt.subplot(339)

b_carat.clarity.value_counts().plot.pie(autopct='%.1f%%', radius=4, colors=colors_a)

plt.show()
plt.figure(figsize=(9, 6))

sns.barplot(x= "clarity", y="price", data = b_carat, palette=("pastel"))

sns.barplot(x= "clarity", y="price", data = s_carat)

plt.show()
colors = [ '#ff6666','#ffcc99', '#c2c2f0',  '#fec8d8', '#f9f9aa', '#66b3ff', '#99ff99']

colors_b = ['#66b3ff',  '#fec8d8', '#99ff99', '#ff6666','#c2c2f0', '#ffcc99', '#f9f9aa']



plt.subplot(232)

diamonds.color.value_counts().plot.pie(autopct='%.2f%%', radius=1.5, colors=colors)

plt.subplot(234)

s_carat.color.value_counts().plot.pie(autopct='%.1f%%', radius=2, colors=colors)

plt.subplot(236)

b_carat.color.value_counts().plot.pie(autopct='%.1f%%', radius=2, colors=colors_b)

plt.show()
plt.figure(figsize=(8, 5))

sns.barplot(x= "color", y="price", data = b_carat,  palette=("pastel"))

sns.barplot(x= "color", y="price", data = s_carat)

plt.show()
# 1.75 캐럿 이상의 다이아몬드의 가격분포

plt.hist(b_carat["price"], bins=20)

plt.show()
b_carat["count"]=1

pv = b_carat.pivot_table("count", ["color", "cut"], "clarity" , aggfunc=np.sum)



plt.figure(figsize=(10, 10))

sns.heatmap(pv, cmap='YlGnBu')

plt.show()
plt.figure(figsize=(10, 20))

plt.subplot(311)

sns.boxenplot(data = b_carat.sort_values(by="color"), x="color", y="price", color ="r")

plt.subplot(312)

sns.boxenplot(data = b_carat, x="clarity", y="price", color ="b")

plt.subplot(313)

sns.boxenplot(data = b_carat.sort_values(by="cut"), x="cut", y="price", color ="g")

plt.show()
b_carat[(b_carat["clarity"]=="IF") & (b_carat["color"]=="J")]
select_dia = b_carat.copy()

select_dia["clarity"]=select_dia["clarity"].replace(["IF","VVS1", "VVS2", "I1"], np.NaN)

select_dia["color"]=select_dia["color"].replace(["D","E","F","J"], np.NaN)

select_dia["cut"]=select_dia["cut"].replace("Fair", np.NaN)

select_dia.dropna(axis=0, inplace=True)



sns.catplot(x= "color", y="price", hue="clarity", col="cut", col_wrap=2, kind="point", capsize=.2, data = select_dia)

plt.show()
diamonds_ohe = pd.get_dummies(diamonds, prefix=['cut', 'color', 'clarity'])
# 선형회귀 모델

from sklearn import linear_model

# 학습 데이터와 테스트 데이터를 나눠주는 모듈

from sklearn.model_selection import train_test_split

# 결과 데이터를 평가해주는 모듈

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 데이터, 타겟 구분

dm_x = diamonds_ohe[["carat","depth","table","x","y","z",

                     "cut_Fair", "cut_Good", "cut_Very Good","cut_Premium","cut_Ideal", 

                     "color_D","color_E", "color_F","color_G", "color_H", "color_I","color_J", 

                     "clarity_I1", "clarity_IF", "clarity_SI1", "clarity_SI2", "clarity_VS1", "clarity_VS2", "clarity_VVS1", "clarity_VVS2"]]

dm_y = diamonds_ohe[["price"]]



x_train, x_test, y_train, y_test = train_test_split(dm_x, dm_y, test_size=0.3, random_state=1)
# 객체 생성 / 모델 학습 / 결과 데이터 예측

reg = linear_model.LinearRegression()

reg.fit(x_train, y_train)

predict_result = reg.predict(x_test)
# 모델 평가

print("accuracy: "+ str(reg.score(x_test, y_test)*100) + "%")

print("Mean absolute error: {}".format(mean_absolute_error(y_test, predict_result)))

print("Mean squared error: {}".format(mean_squared_error(y_test, predict_result)))

R2 = r2_score(y_test, predict_result)

print('R Squared: {}'.format(R2))



n=x_test.shape[0]

p=x_test.shape[1] - 1

adjusted_r_squared = 1 - (1 - R2) * ((n - 1)/(n-p-1))

print('Adjusted R Squared: {}'.format(adjusted_r_squared))
