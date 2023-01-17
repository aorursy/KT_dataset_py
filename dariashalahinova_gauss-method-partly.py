# Функция для вывода матрицы

def show_matrix(matrix):

    for item in matrix:

        for value in item:

            print(value, end='\t')

        print()

    print()
# Округление значений элементов матрицы

def round_matrix(matrix):

    for i in range(len(matrix)):

        for j in range(len(matrix) + 1):

            matrix[i][j] = round(matrix[i][j], 2)
# Ищем значение самого большого элемента в столбце(По модулю) и возвращаем номер его строки

def find_row_index(matrix, col_numbder):

    if col_numbder > len(matrix) - 1:

        return

    max_v = abs(matrix[col_numbder][col_numbder])

    max_i = col_numbder

    for i in range(col_numbder, len(matrix)):

        # ищем максимальный элемент

        if abs(matrix[i][col_numbder]) > max_v:

            max_v = abs(matrix[i][col_numbder])

            max_i = i

    return max_i
# Приводим заданный столбец к виду верхней треугольной матрицы

def calculate_column(matrix, col_number):

    x = matrix[col_number][col_number]

    for i in range(col_number + 1, len(matrix)):

        # Получаем значегте коэфициента на который будем домножать строку, перед тем как прибавить

        coef_val = matrix[i][col_number]/x

        # Определяем знак коэфициента(+/-)

        if coef_val < 0:

            coef_val = abs(coef_val)

        else:

            coef_val = coef_val * -1

        # Складываем строку, которую хотим привести к нужному виду со строкой в которой самый большой элемент в столбце

        # умноженной на коэфициент

        for j in range(len(matrix[i])):

            matrix[i][j] = matrix[i][j] + (matrix[col_number][j] * coef_val)
# Ищем значение наших неизветснх

def find_answer(matrix):

    # Вывод матрицы и названия неизвестных рядом с ней

    for i in range(len(matrix)):

        print(matrix[i],  'x' + str(i + 1), sep='\t',)

    answers = []

    # Цикл для прохода по матрице начиная с последней строки

    for i in range(len(matrix) - 1, -1, -1):

        sum = 0

        answers_index = 0

        # Цикл для доступа к элементами матрица необходимым для решения уровнения

        for j in range(len(matrix) - 1, i -1, -1):

            # Решения последней строки матрицы

            if i == len(matrix) - 1:

                answers.append(matrix[i][j] / matrix[i][len(matrix)])



            # Решение для остальных строк матрицы

            else:

                # Просчитываем значения эл в строке, для которых уже знаем ответ и суммируем их.

                if j > i:

                    sum += answers[answers_index] * matrix[i][j]

                    answers_index += 1

                #Если не знаем ответ, то получаем его вынося сумму за знак равенства

                # и деля на это число оставшийся элемент с троке

                else:

                    # Длюавляем ответ, если он равен 0

                    if (matrix[i][len(matrix)] - sum) == 0:

                        answers.append(0)

                    #Добавляем ответ

                    else:

                        answers.append(matrix[i][j]/ (matrix[i][len(matrix)] - sum))

                    # answers_index += 1

    #Переворачиваем список с ответами, чтобы получить правильную последовательность ответов

    answers.reverse()

    #Выводим ответы

    for i in range(len(matrix)):

        print('x' + str(i + 1), '=', answers[i])
matrix = [[10, -7, 0, 7],

          [-3, 2, 6, 4],

          [5, -1, 5, 6]]

print('Матрица')

show_matrix(matrix)



for i in range(len(matrix)):

    if i >= len(matrix) - 1:

        continue

    # получаем индекс самого большого элемента в столбце

    ind = find_row_index(matrix, i)

    # Меняем местами необходимые столбцы

    matrix[i], matrix[ind] = matrix[ind], matrix[i]

    # Приводим нужных столбце к виду треугольной матрицы

    calculate_column(matrix, i)

    # Округляем полученные значения

    round_matrix(matrix)



print('Матрица после приведения к верхнему треугольному виду')

show_matrix(matrix)

print('Вывод неизвестых и ответов')

find_answer(matrix)