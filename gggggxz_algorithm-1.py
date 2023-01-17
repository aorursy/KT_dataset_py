import time
n, m = 5, 2

value = list(range(1, n + 1))

print(value)
pos = 0

print(value[pos])

pos = pos + m

print(value[pos])
pos = 4

pos = (pos + m) % n  # 模运算，即取余

print(value[pos])
value.pop(pos)  # 删除列表元素

print(value)
def Joseph_arr(n, m):

    """

    @param n: 

    @param m: 

    @return: 

    """

    value = list(range(1, n + 1))

    pos = n - 1

    for i in range(n - 1):

        pos = (pos + m) % len(value)

        value.pop(pos)

        pos = pos - 1

    return value[0]
Joseph_arr(5, 2)
# 测试运行时间

start = time.time()

Joseph_arr(200000, 100)

end = time.time()

print(end - start)
a_list = []

start = time.time()

for i in range(100000):  # 重复往a_list结尾处添加元素100000次

    a_list.append(0)

end = time.time()

print(end - start)
start = time.time()

for i in range(100000):  # 重复删除a_list的第0个元素100000次

    a_list.pop(0)

end = time.time()

print(end - start)
value = list("abcdefghij")  # 链表元素

n = 10

print(value)
pre = [0] * 10  # 前向链

nxt = [0] * 10  # 后向链

for i in range(n):

    pre[i] = (i - 1) % n

    nxt[i] = (i + 1) % n
def print_list(value, pre, nxt, pos):

    """

    从pos处开始，输出链表中的所有元素

    @param value: 链表元素值

    @param pre: 链表元素的前向链

    @param nxt: 链表元素的后向链

    @param pos: 起始位置

    @return:

    """

    start = pos  # 用start保存起始位置

    result = []

    while True:

        result.append(value[pos])

        pos = nxt[pos]

        if pos == start:  # 如果下一个元素的下标是start，说明已经遍历完整个环

            break

    print(result)



print_list(value, pre, nxt, 0)

print_list(value, pre, nxt, 3)
# pos从位置0往后移动三步

pos = 0

for i in range(3):

    pos = nxt[pos]

print_list(value, pre, nxt, pos)
# 删除pos处的元素

left, right = pre[pos], nxt[pos]  # value[pos]两侧元素的下标

pre[right] = left

nxt[left] = right

print_list(value, pre, nxt, left)  # 此时pos处的元素已经从链表中删去，所以从left处开始输出
def Joseph_linked_list(n, m):

    """

    @param n: 

    @param m: 

    @return: 

    """

    value = list(range(1, n + 1))

    pre = [0] * n

    nxt = [0] * n

    for i in range(n):

        pre[i] = (i - 1) % n

        nxt[i] = (i + 1) % n

    pos = n - 1

    for i in range(n - 1):

        for j in range(m):

            pos = nxt[pos]

        left, right = pre[pos], nxt[pos]

        pre[right] = left

        nxt[left] = right

        pos = left

    return value[pos]
Joseph_linked_list(5, 2)
start = time.time()

Joseph_linked_list(200000, 100)

end = time.time()

print(end - start)