# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
class Poly: #代表多项式的类
    
    content = []
    hp = 0
    start = 0
    end = 0
    #定义实例化方法
    def __init__(self,_content,from_np = False): #后面我发现from_np这个选项完全没有存在的必要
        if not from_np:
            self.hp = _content[-1][1]
            self.content = np.zeros((self.hp+1,2),dtype = "int32")
            self.content[:,1] = range(self.hp+1)
            for i in _content:
                self.content[i[1],0] = i[0]
        else:
            self.content = _content
            self.hp = _content[-1][1]
        
        self.end = self.hp
        return
    #定义打印方法
    def __str__(self):
        stri = ""
        for i in self.content:
            if i[0] != 0:
                stri += str(i[0]) + "*X^" + str(i[1]) + " + "
        stri = stri[:len(stri)-2]
        return stri
    #迭代
    def __iter__(self):
        return self
    #定义迭代方法
    def __next__(self):
        if self.start <= self.end:
            ret = self.start
            self.start += 1
            return Poly(self.content[ret,:].reshape(1,2),from_np=True)
        else:
            self.start = 0
            raise StopIteration
def poly_add(po1,po2): #这个方法里有很多无效代码
    re_hp = po1.hp if po1.hp > po2.hp else po2.hp
    low_hp = po1.hp if po1.hp < po2.hp else po2.hp
    re_poly =np.zeros((low_hp + 1,2),dtype = "int32")
    for i in range(low_hp+1):
        re_poly[i][0] = po1.content[i][0] + po2.content[i][0]
        re_poly[i][1] = po1.content[i][1]
    loger = po1 if po1.hp > po2.hp else po2
    re_poly = np.concatenate((re_poly,loger.content[low_hp+1:re_hp+1,:]))
    return Poly(re_poly,from_np=True)
def poly_multiply(po1,po2): #这个方法完全是绿皮代码瞎鸡脖实现的
    origin = Poly([[0,0]])
    sum_pool = []
    for i in po1:
        min_pool = []
        for j in po2: #穷举就完事了
            ar = [i.content[0][0] * j.content[0][0],i.content[0][1] + j.content[0][1]]
            min_pool.append(np.array(ar,dtype = "int32").reshape(1,2))
        sum_pool.append(Poly(np.concatenate(min_pool)))
    for i in sum_pool:
        origin = poly_add(origin,i)
    return origin
            
#示例
po1 = Poly([[1,1],[2,3],[3,4],[4,5],[3,6]])
po2 = Poly([[2,3],[3,4],[4,5]])
print(poly_add(po1,po2))
