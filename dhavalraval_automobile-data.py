import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

auto_data = pd.read_csv('../input/automobile-dataset/Automobile_data.csv')
auto_data.head()

print(auto_data[['body-style','num-of-doors']].head(10))
print(auto_data[['body-style','num-of-doors']].count())
#prices = auto_data.groupby('body-style').mean()['num-of-doors']
#print(prices)

auto_data.groupby(['num-of-doors','body-style']).count()



auto_data['fuel-type'].value_counts()
car = auto_data.groupby(['make','fuel-type'])
rpm = car['make','horsepower'].count()
rpm


auto_data.sort_values(["horsepower" , "peak-rpm"], axis=0, ascending=False, inplace=True)
auto_data.head()


auto_data = auto_data [['make','price']][auto_data.price==auto_data['price'].max()]
auto_data


auto_data = auto_data.sort_values(by=['price', 'horsepower'], ascending=False)
auto_data.head(5)


car = auto_data.groupby('make')
rpm = car['make','peak-rpm'].mean()
rpm


cars = auto_data.groupby('make')
prices = cars['make','price'].max()
prices


auto_data['make'].value_counts()

g = sns.factorplot("make", data=auto_data, aspect=1.5, kind="count", color="b")
g.set_xticklabels(rotation=90)
car = auto_data.groupby('make')
audi = car.get_group('alfa-romero')
audi

