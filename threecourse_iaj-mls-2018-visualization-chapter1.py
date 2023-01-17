print("Hello, World!")
print("日本語も使えます")
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

x = np.arange(0.0, 6.0, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()
# 値の宣言、代入
x = 1
print(1)

# 複数の変数に一度に代入することもできる
x1, x2 = 1, 2
print(x1, x2)
# 整数型の扱い
x = 2
y = 4
z = x * y
print(x, y, z) # printを使うと情報の出力ができる
# 小数の扱い
x = 1.5
y = 2.0
z = x * y
print(x, y, z)
# 文字列の扱い
pp = "PP" # 文字列はダブルクォーテーションで囲む
ap = 'AP' # シングルクォーテーションでもOK
ppap = pp + ap
print(ppap)
# ブール型の扱い
f1 = True
f2 = False
print(f1)
print(f2)
# 様々な演算
x = 8
y = 3
print(x + y)
print(x - y)
print(x * y)
print(x / y) # 除算（余りを切り捨てない）、他のプログラミング言語では整数同士の除算のときに余りを切り捨てることがあるので注意
print(x // y) # 除算（余りを切り捨てる）

f1 = 1 == 1
f2 = 1 != 1
f3 = 1 < 2
f4 = 1 > 2
print(f1, f2, f3, f4)
# 複合演算子
x = 10
print(x)
x += 1
print(x)
x *= 2
print(x)
x /= 3
print(x)
# 型の確認と変換
x = 1.0
y = "10"
print(type(x))
print(type(y))

st_x = str(x)
int_y = int(y)
float_y = float(y)
print(st_x, type(st_x))
print(int_y, type(int_y))
print(float_y, type(float_y))
# リストの定義
lst = [0, 10, 20, 30, 40]
print(lst)

# リストからの値の取り出し
# 添字は0から始まることに注意
print(lst[0], lst[2])

# リストへの値の追加
lst.append(50)
print(lst)

# リストの連結
lst1 = [0, 100]
lst2 = [200, 300]
lst3 = lst1 + lst2
print(lst3)

# リストへの値の代入
lst[0] = 1000
print(lst)

# リストのスライス
# 1:3と指定した場合、添字が1,2のものが取り出される
print(lst[1:3])

# リストの長さを出力 - len関数
print(len(lst))
# タプルの定義
tpl = (0, 10, 20, 30, 40)
print(tpl)

# タプルからの値の取り出し
print(tpl[0], tpl[2])

# タプルへの値の追加・・・はできない
# タプルの連結はできるが、あまりしない
# タプルへの値の代入・・・はできない
# タプルのスライスもできるが、あまりしない

# タプルの長さを出力 - len関数
print(len(tpl))
# ディクショナリの定義
dic = {}  # 空のディクショナリを作る
print(dic)
dic = {"a":1, "b":2}  # 初期値の入ったディクショナリを作る
print(dic)

# 値の取得
print(dic["a"])

# 値の変更
dic["a"] = 10
print(dic["a"])

# キー・値の追加
dic["c"] = 3
print(dic["c"])

# キー・値の削除
dic.pop("c")

# キーの存在判定
print("a" in dic)
print("c" in dic)
# for文
lst = [0, 10, 20]

for e in lst:
    print(e)

# range関数を使って列挙する    
for i in range(len(lst)):
    print(i, lst[i])
    
# enumerate関数を使って列挙する    
for i, e in enumerate(lst):
    print(i, e)
# if文
a = 2
if a == 1:
    print("aは1です")
elif a == 2:
    print("aは2です")
else:
    print("aは1でも2でもない")
    
# if文とfor文の組み合わせ
lst = [0, 1, 2, 3, 4, 5]

for e in lst:
    if e % 2 == 0:
        print(e, "is even")
    else:
        print(e, "is odd")
# リスト内包表記によるリストの作成
lst = [x * 2 for x in range(5)]
print(lst)

# for文を使っても同じものが作れます
lst2 = []
for x in range(5):
    lst2.append(x * 2)
    
print(lst2)
# 加算した値を返す関数の定義（戻り値を返す関数）
def add(a, b):
    return a + b

# 修飾してprintする関数の定義（戻り値が返さない関数）
def print_decorate(x):
    print("** " + str(x) + " **")

# 関数の使い方    
c = add(10, 20)
print(c)
st = "dog"
print_decorate(st)
m = 10

def multiply(a):
    _z = a * m
    return _z

print(multiply(5))
# print(_z) # エラーとなる
# YOUR CODE GOES HERE
# YOUR CODE GOES HERE