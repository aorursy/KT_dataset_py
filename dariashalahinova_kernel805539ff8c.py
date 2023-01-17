def show_matrix(matrix):

    for item in matrix:

        for value in item:

            print(value, end='\t')

        print()

    print()
m = [[1,1,1,6],

          [1,-1,2,5],

          [2,-1,-1,-3]]

show_matrix(m)


for col in range(len(m[0])):

    for row in range(col+1, len(m)):

        r = [(rowValue * (-(m[row][col] / m[col][col]))) for rowValue in m[col]]

        m[row] = [sum(pair) for pair in zip(m[row], r)]

        # print('first output', col=)

        show_matrix(m)

# now backsolve by substitution

ans = []

m.reverse() # makes it easier to backsolve

show_matrix(m)

for sol in range(len(m)):

        if sol == 0:

            ans.append(m[sol][-1] / m[sol][-2])

        else:

            inner = 0

            # substitute in all known coefficients

            for x in range(sol):

                inner += (ans[x]*m[sol][-2-x])

            # the equation is now reduced to ax + b = c form

            # solve with (c - b) / a

            show_matrix(m)

            ans.append((m[sol][-1]-inner)/m[sol][-sol-2])

ans.reverse()
ans