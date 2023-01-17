#参考サイト : https://pythondatascience.plavox.info/matplotlib/%e6%a3%92%e3%82%b0%e3%83%a9%e3%83%95
import numpy as np
import matplotlib.pyplot as plt
 
left = np.array([1, 2, 3, 4, 5])
height = np.array([100, 200, 300, 400, 500])
plt.bar(left, height)
plt.bar(left, height, width=1.0)
plt.bar(left, height, color="#FF0000", linewidth=0)
plt.bar(left, height, color="#00FF00", edgecolor="#0000FF", linewidth=4)
plt.bar(left, height, color="#0000FF", align="center")
labels = ["UK", "Germany", "Japan", "China", "USA"]
plt.bar(left, height, tick_label=labels, color="#FF00FF", align="center")
plt.bar(left, height, tick_label=labels, color="#FFFF00", align="center")
plt.title("This is a title")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.grid(True)
plt.bar(left, height, tick_label=labels, color="#00FFFF", align="center", xerr=0.5, ecolor="red")
yerr = np.array([10, 20, 30, 40, 50])
plt.bar(left, height, tick_label=labels, color="blue", yerr=yerr, ecolor="black")
a = np.arange(1,20)
b = a * a
plt.bar(a, b, color="blue")
plt.title("This is a title")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.grid(True)