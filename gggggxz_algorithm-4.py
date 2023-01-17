def check(vis):

    s1, s2 = 0, 0

    for v in vis:

        if v == 1:

            s1 += 1

        else:

            s2 += 1

        if s1 - s2 < 1:

            return False

    return True
print(check([1, 1, 1, 1, 0, 0, 0]))

print(check([1, 1, 1, 0, 1, 0, 0]))

print(check([1, 1, 1, 0, 0, 1, 0]))

print(check([1, 1, 1, 0, 0, 0, 1]))

print(check([1, 1, 0, 1, 0, 0, 1]))
def find(vis, depth):

    if depth == 5:

        global num

        num += 1  # 变量num记录check函数的执行次数

        if check(vis):  # 递归终止条件

            ans = []

            for i, v in enumerate(vis):

                if v == 1:

                    ans.append(i + 1)

            results.add(tuple(ans))

        return

    start, cnt = 0, 0

    """

    while start < 7 and cnt < depth - 1:

        if vis[start]:

            cnt += 1

        start += 1

    """

    for i in range(start, 7):

        if vis[i] == 0:

            vis[i] = 1

            find(vis, depth + 1)

            vis[i] = 0
vis = [0] * 7

results = set()

num = 0

find(vis, 1)

print('check函数执行次数：', num)

print(results)
def permutation(arr, vis, depth):

    n = len(vis)

    if depth == n + 1:

        print(arr)

        return

    for i in range(len(vis)):

        if vis[i] == 0:

            arr.append(i + 1)

            vis[i] = 1

            permutation(arr, vis, depth + 1)

            arr.pop()

            vis[i] = 0
n = 3

arr = []

vis = [0] * n

permutation(arr, vis, 1)