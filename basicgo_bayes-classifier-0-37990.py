import pandas, numpy

import os

import matplotlib.pyplot as plot

from tqdm import tqdm
data_train = pandas.read_csv(os.path.join('/', 'kaggle', 'input', 'ozon-masters-2020', 'gtrain.csv'))

data_test = pandas.read_csv(os.path.join('/', 'kaggle', 'input', 'ozon-masters-2020', 'gtest.csv'))
nsteps_indices = []

test_nsteps_indices = []



for k in range(1, 6):

    nsteps_indices.append(numpy.array(data_train[data_train.steps == k].index))

    test_nsteps_indices.append(numpy.array(data_test[data_test.steps == k].index))
N, N2 = 22, 484



SIZE = (data_train.shape[0], N, N)



y_train = data_train.iloc[:, 2+N2:2+2*N2].to_numpy().reshape(SIZE)

x_train = data_train.iloc[:, 2:2+N2].to_numpy().reshape(SIZE)



TSIZE = (data_test.shape[0], N, N)



x_test = data_test.iloc[:, 2:2+N2].to_numpy().reshape(TSIZE)

y_test = numpy.empty(TSIZE, dtype = 'float32')
magic_square = numpy.mean(y_train, axis = 0)



plot.imshow(magic_square)

plot.title('Магический квадрат')

plot.show()
magic_square_brer_train = numpy.mean(x_train, axis = 0)

magic_square_brer_test = numpy.mean(x_test, axis = 0)



f, axes = plot.subplots(1, 2)



axes[0].imshow(magic_square_brer_train.reshape(N, N))

axes[0].set_title('Брат магического квадрата')



axes[1].imshow(magic_square_brer_test.reshape(N, N))

axes[1].set_title('Его близнец')



plot.show()
def bayes_predict(image, steps_indices, nsteps_range = range(5)):

    ONE_CP = numpy.empty((N, N, N, N), dtype = 'float32')

    ZERO_CP = numpy.empty((N, N, N, N), dtype = 'float32')

    result = numpy.empty((N, N), dtype = 'float32')



    for nsteps_ in nsteps_range:

        indices = steps_indices[nsteps_]



        nstep_x_train = x_train[nsteps_indices[nsteps_]]

        nstep_y_train = y_train[nsteps_indices[nsteps_]]



        items2pred = image[indices]



        # Для заданного числа шагов формируем матрицы условных вероятностей.

        with tqdm(total = N2, desc = 'Число шагов = %d -> Подготовка' % (nsteps_ + 1), bar_format = "{desc}:   [ осталось: {remaining}; прошло: {elapsed} ] {percentage:3.0f}%|{bar}") as pbar:

            for i in range(N):

                for j in range(N):

                    ZERO_CP[i, j, :] = numpy.mean(nstep_x_train[nstep_y_train[:, i, j] == 0], axis = 0)

                    ONE_CP[i, j, :] = numpy.mean(nstep_x_train[nstep_y_train[:, i, j] == 1], axis = 0)

                    ######################################

                    pbar.update(1)



        # Предсказываем...

        with tqdm(total = items2pred.shape[0], desc = 'Число шагов = %d -> Предсказание' % (nsteps_ + 1), bar_format = "{desc}: [ осталось: {remaining}; прошло: {elapsed} ] {percentage:3.0f}%|{bar}") as pbar:

            for k in range(items2pred.shape[0]):



                item2pred = items2pred[k]

                inverted = item2pred ^ 1



                for i in range(N):

                    for j in range(N):

                        c = (ONE_CP[i, j] * item2pred + inverted * (1 - ONE_CP[i, j])).prod()

                        d = (ZERO_CP[i, j] * item2pred + inverted * (1 - ZERO_CP[i, j])).prod()



                        v = magic_square.item(i, j)



                        # В числителе намеренно не умножаем на magic_square.itemset(i, j), поскольку так медленнее.

                        result.itemset(i, j, c / (v * c + (1 - v) * d))



                # Умножаем сразу всю матрицу здесь.

                result *= magic_square



                y_test[indices[k], :] = result

                ######################################

                pbar.update(1)
bayes_predict(x_test, test_nsteps_indices)
columns = []



for k in range(N2):

    columns.append(f'y_{k}')

    

answer = numpy.int32(y_test + .5).reshape(data_test.shape[0], N2)



submission = pandas.DataFrame(answer, columns = columns)



submission.insert(0, 'id', data_test['id'].values)

submission.to_csv('submission.csv', index = False)