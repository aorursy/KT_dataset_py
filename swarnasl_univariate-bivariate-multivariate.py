import pandas as pd

d = {"Height":[164,167.3,170,174.2,178,180]}
Height_of_the_students = pd.DataFrame(d)
Height_of_the_students
Height_of_the_students.describe()
Height_of_the_students.info()
import matplotlib.pyplot as plt

Height_of_the_students.plot()
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x = 'Height', data = Height_of_the_students)
sns.boxplot(y = Height_of_the_students, palette = 'Reds')
sns.violinplot(y = Height_of_the_students, palette = 'Oranges')
dict = {"Temp":[20,25,27,29,31,35], "Sales":[2000,2300,2600,2800,3000,3400]}

Temp_data = pd.DataFrame(dict)
Temp_data
Temp_data.describe()
Temp_data.info()
Temp_data.plot()
Temp_data.hist()
sns.barplot(x = 'Temp', y = 'Sales', data = Temp_data)
sns.stripplot(x = 'Sales', data = Temp_data)
dict = {"No.of_Rooms":[2,3,3.5,4],"Floors":[0,2,3,5],"Area(S.Ft)":[900,1100,1500,2100],"Price($)":[4000,600,900,1200]}

house_price = pd.DataFrame(dict)
house_price
house_price.info()
house_price.describe()
house_price.hist()
plt.show()
import seaborn as sns

sns.barplot(y = 'Floors', x = 'No.of_Rooms', data = house_price)