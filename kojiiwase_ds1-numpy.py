result_list=[i*10 for i in range(0,4)]
result_list

x=lambda path :path.split('/')[-1]

x('/home/user/Desktop/image.jpg')
def print_list(*args):

  for elem in args:

    print(elem)
print_list('java','python','c++')
def print_dict(**kwargs):

  print(kwargs)
print_dict(key='value')
def print_dict(**kwargs):

  # kwargs.get(key) でそのkeyのvalueを取り出すことができます．

  param1=str(kwargs.get('param1'))

  # .get() にもう一つ引数を渡すことでデフォルトの値を指定できます

  param2=str(kwargs.get('param2','defalut_value'))

  # valueが指定されていないので、Noneを返す

  param3=str(kwargs.get('param3'))



  print('param1 is {}'.format(param1))

  print('param2 is {}'.format(param2))

  print('param3 is {}'.format(param3))
print_dict(param1='param1_value')
import numpy as np

py_list=[[1,2,3],[4,5,6,],[7,8,9]]

py_list
np_list=np.array([[1,2,3],[4,5,6],[7,8,9]])

np_list
np.arange(0,5)
np.arange(0,5,2)
np.arange(5)

# これも「n以上m未満」の法則適用される
# np.linspace(start,stop,num=50)

# startからstopまでの値をnum等分した値がarrayで返されます。50はデフォると

np.linspace(1,2,20)

# stopの値を含むことに注意しましょう．

# np.copy

# arrayをコピーする（値渡し）

# NumPy Arrayは関数に渡すとき，参照渡しになることに気をつけましょう．

# 値渡しにしたいときに.copy()をよく使います





ndarray = np.arange(0, 5)

ndarray_copy = ndarray.copy()

print("original array's id is {}".format(id(ndarray)))

print("copied array's id is {}".format(id(ndarray_copy)))

 

#changing original array

ndarray[:] = 100

print('original array:\n', ndarray)

print('copied array:\n', ndarray_copy)
# 乱数生成

# np.random(0から1の間でランダムな数字を作る)



random_float = np.random.rand()

random_1d = np.random.rand(3)

random_2d = np.random.rand(3, 4)



print('random_float: {}\n'.format(random_float))

print('random_1d: {}\n'.format(random_1d))

print('random_2d: {}\n'.format(random_2d))



# あるデータ分布からランダムサンプリングするときなんかに使えます．
np.random.randn()

np.random.randn(3,4)
np.random.randn(3,4)
np.random.randint(10, 50, size=(2, 4, 3))
array=np.arange(0,10)

new_shape=(2,5)

reshaped_array=array.reshape(new_shape)

reshaped_array
# 適当に標準正規分布を作成

normal_dist_mat = np.random.randn(5, 5)

print(normal_dist_mat)
# 最大値

print(normal_dist_mat.max())

# 最大値のindex

print(normal_dist_mat.argmax())

# 平均値

print(normal_dist_mat.mean())

#標準偏差

print(normal_dist_mat.std())

# 中央値　他と書き方違うので注意

print(np.median(normal_dist_mat))
print(normal_dist_mat)

print('axis=0> {}'.format(normal_dist_mat.max(axis=0)))

print('axis=1> {}'.format(normal_dist_mat.max(axis=1)))
# np.expand_dims(ndarray, axis)

# rankを一つ追加します．axis=0なら一つ目の次元を，axis=-1なら最後の次元を追加します．



# 使うのは大抵axis=0かaxis=-1です．



ndarray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

expanded_ndarray = np.expand_dims(ndarray, axis=0)

print(expanded_ndarray.shape)

print(expanded_ndarray)

# []が一つ外側に増えてる





# np.squeeze(ndarray)

# squeeze＝絞る







squeezed_expanded_ndarray = np.squeeze(expanded_ndarray)

squeezed_expanded_ndarray.shape







# flatten()

# ndarrayを一列にします．行列構造を持つ必要がなくなったりしたら使います．



flatten_array = ndarray.flatten()

print('flatten_array:\n{}'.format(flatten_array))

print('ndarray:\n{}'.format(ndarray))
# np.save(‘ファイルパス’, ndarray)



ndarray = np.array([

    [1, 2, 3, 4],

    [10, 20, 30, 40],

    [100, 200, 300, 400],

])

np.save('saved_numpy', ndarray)



# jupyterの左側に保存されている

# 拡張子はつけてもつけなくてもOK！



loaded_numpy = np.load('saved_numpy.npy')

loaded_numpy