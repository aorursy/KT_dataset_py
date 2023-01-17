[0, 1, 2]
len([0, "あ", 2])
[0, 1, "あ", 2, 2, 2, 3].count(2)
[0, 1, 2, 3, 4, 5][0:4]
[0, 1, 2, 3, 4, 5][0:-1]
[0, 1, 2, 3, 4, 5][0:]
[0, 1, 2, 3, 4, 5][::2]
type(range(100))
for i in range(3):

    print(i)
for i in range(0, 4, 3):

    print(i)
list(range(3))
input_text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."



input_text = input_text.replace(".", "") #  . を消す

words = input_text.split()

result = {}

for i, word in enumerate(words):

    if i+1 in [1, 5, 6, 7, 8, 9, 15, 16, 19]:   

        key = word[0]

        value = i

    else:

        key = word[:2]

        value = i

    

    result[key] = value

        

result
word1 = "パトカー"

word2 = "タクシー"



result = ""

for ch1, ch2 in zip(word1, word2):

    result = result + ch1+ ch2 

result
for x, y in zip([1, 2], [2, 3, 5]):

    print(x,y)
xs = [i for i in range(10)]

xs
def double(x):

    return 2 * x
dob_xs = [double(x) for x in xs]

print(dob_xs)
# xsから3の倍数だけを取る

mod3_xs = [x for x in xs if x % 3 == 0]

mod3_xs
# 1. testfunc1: 空リストを用意してappend

def test1(rangelist):

    test_list = []

    for i in rangelist:

        test_list.append(i)





# 3. testfunc3: リスト内包表記      

def test2(rangelist):

    test_list = [i for i in rangelist]
rangelist = range(1,10000000)
%timeit -n 10 -r 3 test1(rangelist)
%timeit -n 10 -r 3 test2(rangelist)
class Cat:

    def __init__(self, name, index):

        self.name = name

        self.index = index

        

    def intro(self):

        print(self.name + "ちゃんは" + str(self.index) + "番目に生まれてきました。")
cat = Cat(name="むぎお", index=2)
cat.name
cat.intro()