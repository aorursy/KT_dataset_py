import numpy as np # linear algebra

from matplotlib import pyplot as plt
x = np.arange(0.0, 8, 0.01)

y1 = 0*x # >=0

y2 = 3-0.6*x # <=0

y3 = 2 + x # <=0

y4 = 5-10/7*x # <=0

y5 = 1 - x # >=0



#we consider two conditions instead of five: 

#that the upper bound must be etermined by the conditions

#that y is less then 0, and the lower bound is greater then 0.

ymax = np.minimum(np.minimum(y2, y3), y4)

ymin = np.maximum(y1, y5)

#мы рассматриваем два условия вместо шести:

#что верхняя граница задается условием, что y меньше 0,

#а нижнее услоиве задается условием, что y больше 0.



plt.ylim(0, 5)



y=np.arange(0.0, 5, 0.01)

plt.plot(0*y, y, label = "x1>=0")

plt.plot(x, y1, label = "x2>=0")

plt.plot(x, y2, label = "3x1+5x2<=15")

plt.plot(x, y3, label = "-x1+x2<=2")

plt.plot(x, y4, label = "10x1+7x2<=35")

plt.plot(x, y5, label = "x1+x2>=1")

plt.fill_between(x, ymin, ymax, color='grey', alpha=0.5, label="область допустимых решений")

plt.grid()

plt.legend()

plt.show()





plt.ylim(0, 5)

y=np.arange(0.0, 5, 0.01)

plt.plot(0*y, y, label = "x1>=0")

plt.plot(x, y1, label = "x2>=0")

plt.plot(x, y2, label = "3x1+5x2<=15")

plt.plot(x, y3, label = "-x1+x2<=2")

plt.plot(x, y4, label = "10x1+7x2<=35")

plt.plot(x, y5, label = "x1+x2>=1")

plt.fill_between(x, ymin, ymax, color='grey', alpha=0.5, label="domain")

plt.grid()

plt.legend()

plt.show()
#we arrange the lines of the same values of the function on the graph along with domain

#расположим линии уровня функции на графике вместе с областью допустимых значений.



x = np.arange(0.0, 8, 0.01)

function = dict()

function[1]=1-x

function[2]=2-x

function[3.25]=3.25-x

function[3.5]=3.5-x

function[3+28/29]=3+28/29-x



plt.ylim(0, 5)





#also plot the corner points

#также разметим угловые точки



plt.scatter([0, 1, 0, 0.625, 70/29, 3.5],[1, 0, 2, 2.625, 45/29, 0])





plt.plot(x, function[1], label = "x1+x2=1")

plt.plot(x, function[2], label = "x1+x2=1")

plt.plot(x, function[3.25], label = "x1+x2=3.25")

plt.plot(x, function[3.5], label = "x1+x2=3.5")

plt.plot(x, function[3+28/29], label = "x1+x2=115/29")

plt.grid()





plt.ylim(0, 5)

plt.fill_between(x, ymin, ymax, color='grey', alpha=0.5, label="область ограниченная условиями")

plt.legend()

plt.show()



plt.ylim(0, 5)



plt.scatter([0, 1, 0, 0.625, 70/29, 3.5],[1, 0, 2, 2.625, 45/29, 0])

plt.plot(x, function[1], label = "x1+x2=1")

plt.plot(x, function[2], label = "x1+x2=1")

plt.plot(x, function[3.25], label = "x1+x2=3.25")

plt.plot(x, function[3.5], label = "x1+x2=3.5")

plt.plot(x, function[3+28/29], label = "x1+x2=115/29")

plt.grid()





plt.ylim(0, 5)

plt.fill_between(x, ymin, ymax, color='grey', alpha=0.5, label="domain")

plt.legend()

plt.show()
x = np.arange(0.0, 8, 0.01)



plt.ylim(0, 5)



#again plot the corner points and the domain intersection with the line of the maximal value

#вновь отметим угловые точки и пересечение области допустимых решений с линией максимального уровня

plt.scatter([0],[1], label="(0,1)")

plt.scatter([1],[0], label="(1,0)")

plt.scatter([0],[2], label="(0,2)")

plt.scatter([0.625],[2.625], label="(0.625,2.625)")

plt.scatter([70/29],[45/29], label="(70/29,45/29)")

plt.scatter([3.5],[0], label="(3.5,0)")

plt.plot(x, function[3+28/29], label = "x1+x2=115/29")

plt.grid()





plt.ylim(0, 5)

plt.fill_between(x, ymin, ymax, color='grey', alpha=0.5, label="область ограниченная условиями")

plt.legend()

plt.show()



plt.scatter([0],[1], label="(0,1)")

plt.scatter([1],[0], label="(1,0)")

plt.scatter([0],[2], label="(0,2)")

plt.scatter([0.625],[2.625], label="(0.625,2.625)")

plt.scatter([70/29],[45/29], label="(70/29,45/29)")

plt.scatter([3.5],[0], label="(3.5,0)")

plt.plot(x, function[3+28/29], label = "x1+x2=115/29")

plt.grid()





plt.ylim(0, 5)

plt.fill_between(x, ymin, ymax, color='grey', alpha=0.5, label="domain")

plt.legend()

plt.show()