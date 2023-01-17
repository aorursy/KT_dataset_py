import numpy as np

# 输入数据

# 有六个结点ABCDEF

# 出度

out_degree = {'A': 3, 'B': 1, 'C': 1, 'D': 3, 'E': 2, 'F': 1};
# 转移矩阵

a = np.array([[0, 0, 0, 1/out_degree['D'], 0, 0],

             [1/out_degree['A'], 0, 0, 0, 1/out_degree['E'], 0],

             [0, 1/out_degree['B'], 0, 1/out_degree['D'], 1/out_degree['E'], 0],

             [1/out_degree['A'], 0, 0, 0, 0, 1/out_degree['F']],

             [1/out_degree['A'], 0, 1/out_degree['C'], 1/out_degree['D'], 0, 0],

             [1/out_degree['A'], 0, 0, 0, 0, 0]]);

print(a);
# 迭代100次，计算PR值

PR = np.transpose(np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]));

for i in range(100):

    PR = np.dot(a, PR);

print(PR);
import numpy as np

# 输入数据

# 有六个结点ABCDEF

# 出度

out_degree = {'A': 3, 'B': 1, 'C': 1, 'D': 3, 'E': 2, 'F': 1};
# 转移矩阵

a = np.array([[0, 0, 0, 1/out_degree['D'], 0, 0],

             [1/out_degree['A'], 0, 0, 0, 1/out_degree['E'], 0],

             [0, 1/out_degree['B'], 0, 1/out_degree['D'], 1/out_degree['E'], 0],

             [1/out_degree['A'], 0, 0, 0, 0, 1/out_degree['F']],

             [1/out_degree['A'], 0, 1/out_degree['C'], 1/out_degree['D'], 0, 0],

             [1/out_degree['A'], 0, 0, 0, 0, 0]]);

print(a);
# 随机情形

PR = np.transpose(np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]));

N = len(PR);

d = 0.85;

for i in range(100):

    PR = np.transpose((1-d) / N * np.ones(N)) + d * np.dot(a, PR);

# 输出

print('A:', PR[0], '\n B:', PR[1], '\n C:', PR[2], '\n D:', PR[3], '\n E:', PR[4], '\n F:', PR[5])