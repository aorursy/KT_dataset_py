import matplotlib.pyplot as plt

import numpy as np
# 简单的绘图

x = np.linspace(0, 100, 50)

plt.plot(x, 2*x)

plt.show() # 显示图形



plt.plot(2*x) # 如果没有第一个参数 x，图形的 x 坐标默认为数组的索引

plt.show() # 显示图形
x = np.linspace(0, 2 * np.pi, 50)

plt.subplot(1, 2, 1) # （行，列，活跃区）

plt.plot(x, np.sin(x), 'g')

plt.subplot(1, 2, 2)

plt.plot(x, np.cos(x), 'b')

plt.show()
x = np.linspace(0, 2 * np.pi, 50)

plt.plot(x, np.sin(x), 'r', label='Sin(x)')

plt.legend() # 展示图例

plt.xlabel('x label') # 给 x 轴添加标签

plt.ylabel('Y label') # 给 y 轴添加标签

plt.title('Sin function') # 添加图形标题

plt.show()