# функция для вывода матрицы

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
# функция которая возвращает строку матрицы(row) умноженную на число(value)

def multi(row, value):

    new_row = []

    for i in range(len(row)):

        new_row.append(row[i] * value)

    return new_row
# цикл для всех строк матрицы

for i in range(len(m)):

    # цикл для того, чтобы превести значение диагонали к 1 div_value - элемент которых находится в данной строке на диагонали

    div_value = m[i][i]

    for j in range(len(m[i])):

        if div_value == 1:

            continue

        # делю строку на значение элемента, что на диагонали, чтобы получить 1 (-1 -> 1)

        m[i][j] *= 1 / div_value

    show_matrix(m)

    # цикл чтобы привести значения элементо не на диагонали к 0

    for j in range(len(m)):

        # отбрасываю строку матрицы, для которой только что привел значение элемента диагонали к 1.

        if j == i:

            continue

        print(j,m[j], m[j][i], multi(m[i],m[j][i]))

        # получаю строку матрицы умноженную, на элемент, который хочу привести к 0(temp_row)

        temp_row = multi(m[i],m[j][i])

        # отнимаю строку матрицы с строкой temp_row для приведения элемента строки к 0

        for z in range(len(m[j])):

            m[j][z] -= temp_row[z]

show_matrix(m)



        
# цикл для получения ответов

for i in range(len(m)):

        print(m[i][len(m[i]) - 1])