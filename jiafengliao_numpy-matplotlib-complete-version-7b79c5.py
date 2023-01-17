import numpy as np
import matplotlib.pyplot as plt
list=[1,2,3,4,5]
array1 = np.array(list)
array1, type(array1)
array1+2
array1*2
array1**2
list2= [[1,2,3],[4,5,6]]
array2= np.array(list2)
array2
array2 = np.array(list2,dtype= 'float')
array2
array2 = np.array(array2,dtype='int')
array2 = np.array(array2,dtype= 'float')
array2
array2.shape
array2.ndim
array2.size
array2.dtype
array2[:2,:2]
array2%2==0
array2[array2%2==0]
array2[1,1]=np.nan
array2[0,2]=np.inf
array2
index = np.isnan(array2)|np.isinf(array2)
array2[index] =666
array2
array2.mean()
array2.max()
array2.min()
print(array2.max(axis=0)) #这里的axis表示沿着这个axis
print(array2.max(axis=1) )
array2_copy = array2[:,:2]
array2_copy
array2_copy[1,1]=3
array2_copy
array2  #表示在array的copy上进行修改，被copy的array也会被相应的修改
array2 = array2.reshape(3,2)
array2
array2_ravel = array2.ravel()
array2_flatten = array2.flatten()
array2_ravel,array2_flatten
array2_ravel[2]=0   #对ravel进行修改的话原array的值会跟着改变
array2
array2_flatten[2]=7   #对flatten修改的话，原array的值不变
array2
a=np.arange(0,9)  #左闭右开
type(a)
a.reshape(3,3,-1,1)   #参数表示在各个维度的个数
a= np.arange(10,0,-3)  #(开头，结尾，每次+或-什么数)
a
a.reshape(2,2)
a = np.linspace(0,10,11) #(开头，结尾，等差数列的数字个数)
a
a = np.logspace(1,10,num=10,base=10)     #(开头，结尾，中间的数字个数，底)   底的xxx次方  xxx取决于（开头，结尾，中间的数字个数）
a
np.log10(a)
a=np.array([1,2,3])
tile_a = np.tile(a,(2,3))   #铺砖，consider a as a brick
tile_a
repeat_a = np.repeat(a, 3)   #原地重复
repeat_a
a= np.array(['cat','cat','dog','dog','sheep'])
np.unique(a)
np.unique(a,return_counts = True)  
np.random.seed(10)
np.random.rand(3,3)
np.random.randint(0,10,size =(3,3,3))
np.random.choice(['cat', 'dog', 'sheep'],size =10,p=[0.5,0.1,0.4])
a = np.arange(9)
a
np.where(a%2==0)
a = np.arange(4).reshape(2,2)
b = np.arange(6,10).reshape(2,2)

c= np.concatenate([a,b],axis=1)
c.shape
np.save("temp_result",c)  #output c as a .npy format file named "temp_result"
load_d = np.load("temp_result.npy")
load_d
def myfunc(x):
    if x%2 ==0:
        return x/2
    else:
        return x**2

myfunc(2),myfunc(5)
a= np.array([1,2,3])
myfunc_v = np.vectorize(myfunc)   #将func vectorize后可以直接在array上逐位应用
type(myfunc_v)
a= np.arange(6).reshape(2,3)
a
def func(x):
    return (max(x)-min(x))/2
np.apply_along_axis(func,0,a) #沿着xx轴，对xx轴上的value逐个运行func
%matplotlib inline
plt.plot([1,2,3,4])
plt.title("title")
plt.show()
x= np.arange(0.,0.6,0.01)
# y=[i**2 for i in x]
plt.plot(x**2)   #可以不写x,默认x为x的索引
plt.show()
x= np.linspace(-np.pi,np.pi,600)
plt.plot(x,np.sin(x))
plt.plot(x,np.cos(x))
plt.show()
plt.plot(x,np.sin(x),x,np.cos(x))
plt.grid(True)
plt.axis([-2,2,-1,1])#轴坐标的限定
plt.xlim(-1,1)  #同样可以通过xlim和ylim来限定
plt.show()

x= np.linspace(-np.pi,np.pi,600)
plt.plot(x,np.sin(x),label='sin')
plt.plot(x,np.cos(x), color= 'red',label='cos',linestyle="--")#设置label后要用legend才能显示
plt.legend(loc=(1,0))   #左下为0，0   右上为1,1
plt.show()
#通过xticks，yticks来设置轴的刻度
x= np.linspace(-np.pi,np.pi,600)
plt.plot(x,np.sin(x),label='sin')
plt.plot(x,np.cos(x), color= 'red',label='cos',linestyle="--")#设置label后要用legend才能显示
plt.legend(loc=(1,0))   #左下为0，0   右上为1,1
plt.xticks([-3,0,3])   #这里可以用latex格式来添加符号或者公式
plt.savefig("wtf.png")#保存图片
plt.show()

#subplot多个图感觉没啥用，用的时候再查
#x : (n,) array or sequence of (n,) arrays

# 这个参数是指定每个bin(箱子)分布的数据,对应x轴
# bins : integer or array_like, optional
# 这个参数指定bin(箱子)的个数,也就是总共有几条条状图

# normed : boolean, optional
# If True, the first element of the return tuple will be the counts normalized to form a probability density, i.e.,n/(len(x)`dbin)
# 这个参数指定密度,也就是每个条状图的占比例比,默认为1
# color : color or array_like of colors or None, optional
# 这个指定条状图的颜色
# 我们绘制一个10000个数据的分布条状图,共50份,以统计10000分的分布情况
y = np.random.rand(1000)
plt.hist(y,100)  #
plt.show()
x = np.random.rand(200)
y=3*x +4
plt.scatter(x,y,s=y,c=x)  #用scatter来判断数据和结果是否有关系
plt.show()
plt.bar([1,2,3,6],[4,2,6,1])  #在x轴为123的时候y是426
plt.show()
x=[20,20,70]
labels = ["a",'b','c']
plt.pie(x,labels=labels)
plt.show()
x = np.arange(0,4,0.2)
y=np.exp(-x)
e = 0.1*abs(np.random.randn(len(y)))
plt.errorbar(x,y,yerr=e,fmt='.-')
plt.show()

