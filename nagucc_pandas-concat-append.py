import pandas as pd

import numpy as np
#定义资料集

df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])

df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])

df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])



#concat纵向合并

res = pd.concat([df1, df2, df3], axis=0)



#打印结果

print(res)
# 仔细观察会发现上面结果的index是0, 1, 2, 0, 1, 2, 0, 1, 2

# 重置Index

#承上一个例子，并将index_ignore设定为True

res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)



#打印结果

print(res)

# 结果的index变0, 1, 2, 3, 4, 5, 6, 7, 8
# join='outer'为预设值

# 此方式是依照column来做纵向合并，有相同的column上下合并在一起，其他独自的column个自成列，原本没有值的位置皆以NaN填充。



#定义资料集

df4 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])

df5 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])



#纵向"外"合并df4与df5

res = pd.concat([df4, df5], axis=0,sort=False, join='outer')



print(res)
#承上一个例子



#纵向"内"合并df4与df5

# 只有相同的column合并在一起，其他的会被抛弃

res = pd.concat([df4, df5], axis=0, join='inner')



#打印结果

print(res)



#重置index并打印结果

res = pd.concat([df4, df5], axis=0, join='inner', ignore_index=True)

print(res)
# append合并只能做纵向合并，不能横向合并



#定义资料集

df6 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])

df7 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])

df8 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])

s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])



#将df7合并到df6的下面，以及重置index，并打印出结果

res = df6.append(df7, ignore_index=True)

print(res)





#合并多个df，将df7与df8合并至df6的下面，以及重置index，并打印出结果

res = df6.append([df7, df8], ignore_index=True)

print(res)



#合并series，将s1合并至df6，以及重置index，并打印出结果

res = df6.append(s1, ignore_index=True)

print(res)