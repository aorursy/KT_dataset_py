%matplotlib inline

from fastai.basics import *
n=100 # 选择数据点数量
x = torch.ones(n,2)  # 创建一个100行2列的 2D tensor

x[:,0].uniform_(-1.,1) # 用uniform random 来生成每一行的第一列的数值

x[:5] # 展示前5行数值
a = tensor(3.,2); a # 创建一个 1D tensor， 其实是一个vector
y = x@a + torch.rand(n) # 设计x与y的关系函数
plt.scatter(x[:,0], y); # 作图：画出x的第一列值与y的关系图
def mse(y_hat, y): return ((y_hat-y)**2).mean() # 手写MSE函数
a = tensor(-1.,1) # 将 a_1 -> x_1, a_2 -> x_2, 两个x,对应两个a
y_hat = x@a # 用我们假设的线(a),与x，一起来预测y

mse(y_hat, y) # 用MSE来计算损失值
plt.scatter(x[:,0],y) # 画出x_1与y的关系图

plt.scatter(x[:,0],y_hat); # 画出x_与预测值之间的关系图
a = nn.Parameter(a); a # 将 a的值赋给模型参数
def update(): # 开始设计SGD函数

    y_hat = x@a # 构建x, a, y之间关系 = 模型结构

    loss = mse(y, y_hat) # 用MSE计算损失值

    if t % 10 == 0: print(loss) # 每10次循环iteration，打印当前损失值

    loss.backward() # 从损失值，倒推出每个a的gradient

    with torch.no_grad():

        a.sub_(lr * a.grad) # a的更新公式 => a = a - lr*a.grad

        a.grad.zero_()
lr = 1e-1 # 学习率设定在0.1

for t in range(100): update() # SGD函数做100次循环迭代
plt.scatter(x[:,0],y)

plt.scatter(x[:,0],x@a);
from matplotlib import animation, rc

rc('animation', html='jshtml')
a = nn.Parameter(tensor(-1.,1))



fig = plt.figure()

plt.scatter(x[:,0], y, c='orange')

line, = plt.plot(x[:,0], x@a)

plt.close()



def animate(i):

    update()

    line.set_ydata(x@a)

    return line,



animation.FuncAnimation(fig, animate, np.arange(0, 100), interval=20)