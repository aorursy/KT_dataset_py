import matplotlib.pyplot as plt

%matplotlib inline


# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = 'Yoruba', 'Ibo', 'Hausa', 'Tiv', 'Ijaw', 'Igala', 'Others'

sizes = [200, 50, 90, 100, 60, 80, 140]

explode = (0, 0, 0, 0, 0, 0, 0.1)  # only "explode" the last slice (i.e. 'Others')



fig1, ax1 = plt.subplots(figsize=(15,10))

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()