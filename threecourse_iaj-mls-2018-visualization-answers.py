# Q1-1
lst = []
for i in range(20):
    lst.append(i * i)
print(lst)

# lst = [i * i for i in range(20)] でもOK
# Q1-2
def calc_sum(lst):
    ret = 0
    for v in lst:
        ret += v
    return ret
    # sum(lst) やnumpyを使っても計算できる

s = calc_sum(lst)
print(s)
# Q2-1
import math
print(math.sin(0.5))
# Q2-2
f = open("../input/mlsinput/ch2-q1.txt", "r")
lines = f.readlines()
lines = [line.rstrip() for line in lines]
f.close()

lines = [str(int(line) * 2) for line in lines]

f = open("ch2-outq1.txt", "w")
f.write("\n".join(lines) + "\n")
f.close()
# Q3-1
import numpy as np

ary = np.arange(6)
ary1 = np.array([[0, 1, 2],
                 [13, 14, 15]])

ans = ary.reshape(2, 3)
ans[1, :] += 10
print(ary1)
# Q3-2
import pandas as pd
df = pd.read_csv("../input/mlsinput/ch3-2.csv")
df["price"] = np.where(df["product"] == "B", 400, df["price"])
# こちらの書き方でも良い
# df.loc[df["product"] == "B", "price"] = 400
df["sales"] = df["price"] * df["count"]
print(df)

g = df.groupby("product")["sales"].sum()
print(g)
# Q4
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

x = np.arange(100) * 0.1 - 5.0
y = 1.0 / (1.0 + np.exp(-x))

ax.plot(x, y, c="green")
plt.show()