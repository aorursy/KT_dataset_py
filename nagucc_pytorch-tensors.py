# 引入包

import torch
# 生成0张量

x = torch.zeros(5, 3)

print(x)
# 生成未初始化的张量

x = torch.empty(5,3)

print(x)



# 默认数据类型为float32

print(x.dtype)
# 张量中元素的总数

torch.numel(x)