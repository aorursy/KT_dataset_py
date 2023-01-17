def solve(n, p):

    pos = 0

    nxt = 1

    stk = []

    while pos < n:

        if len(stk) > 0 and stk[-1] == p[pos]:  # 栈非空且栈顶元素是下一个期望出栈元素

            stk.pop()

            pos += 1

        elif nxt <= n:  # 仍有元素可以入栈

            stk.append(nxt)

            nxt += 1

        else:

            return "No"

    return "Yes"
print(solve(5, [5, 4, 3, 2, 1]))

print(solve(5, [5, 4, 1, 2, 3]))

print(solve(6, [2, 3, 1, 4, 6, 5]))

print(solve(7, [4, 5, 3, 6, 7, 1, 2]))

print(solve(7, [4, 5, 3, 6, 7, 2, 1]))