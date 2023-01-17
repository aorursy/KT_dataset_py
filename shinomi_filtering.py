import numpy as np

import matplotlib.pyplot as plt

from scipy import linalg as la

from scipy import sparse

import pandas as pd

import time

# Импорт данных

with open ('../input/diploma/_6.csv', "r") as file:

    df = pd.read_csv(file, delimiter = ",")

    df.columns = ((df.columns.str).replace("^ ","")).str.replace(" $","")

    force_y = df['actual_TCP_force_1']

    pose_x = df['actual_TCP_pose_0']

    pose_y = df['actual_TCP_pose_1']

    pose_z = df['actual_TCP_pose_2']
#####################################################################################################

# Функция для отображения графика

#####################################################################################################

def Graph_Display(x,title,label,legend):

    plt.figure(figsize=(15,10))

    plt.title(title)

    plt.grid(True)

    plt.xlabel('Номер измерения')

    plt.ylabel(label)

    plt.plot(x, color='k', label = "F")

    plt.legend(legend, loc="upper right")

    plt.show()



#####################################################################################################

# Функция для настройки параметров графика

#####################################################################################################

def Graph_Display_Settings(title,label):

    plt.figure(figsize=(15,10))

    plt.title(title)

    plt.grid(True)

    plt.xlabel('Номер измерения')

    plt.ylabel(label)
#####################################################################################################

# ФИЛЬТР БОЙДА

#####################################################################################################

all_time_BOYD_all = []

def Boyd_Filter(y,lam):

    start_time_BOYD = time.time()

    # ПАРАМЕТРЫ 

    # минимизация (1/2)||y-x||^2+lambda*||Dx||_1

    ALPHA = 0.01     # параметр поиска обратной линии (0,0.5]

    BETA = 0.5       # параметр поиска обратной линии (0,1)

    MU = 2           # IPM параметр

    MAXITER = 40     # IPM параметр: количество итераций для IPM

    MAXLSITER = 20   # IPM parameter: количество итераций поиска линии

    TOL = 1e-8       # IPM параметр: толерантность

    n = len(y)       # длина сигнала x

    m = n-2          # длина Dx

    D = np.zeros((n-2,n))

    j_1=0

    j_2=1

    j_3=2

    # Получение разностной матрицы 2-го порядка

    for i in range(n-2):

        D[i][j_1] = 1  

        j_1+=1

    for i in range(n-2):

        D[i][j_2] = -2

        j_2+=1

    for i in range(n-2):

        D[i][j_3] = 1

        j_3+=1

    DDT = D.dot(D.T)

    Dy  = (D.dot(y)).T # замена у трансп вместо у

    Dy = np.array([Dy]).T 

    z = np.zeros((m,1)) # двойственная переменная 

    mu1 = np.ones((m,1)) # копия двойственной переменной(1)

    mu2 = np.ones((m,1)) # копия двойственной переменной(2)

    t = 1e-10

    pobj =  'inf'

    dobj =  0

    step =  0.3 # Шаг 

    f1 =  z-lam 

    f2 = -z-lam

    gap_mass = []

    for i in range(0,MAXITER): 

        DTz  = ((z.T).dot(D)).T

        DDTz = D.dot(DTz)

        w  = Dy-(mu1-mu2) # Расчет вектора w

        k = np.linalg.lstsq(DDT,w,rcond=None)[0]

        pobj1 = 0.5*w.T.dot(k) + lam*sum(mu1+mu2) # Двойственная переменная, необходимая для оценки результата #1

        pobj2 = 0.5*(DTz.T).dot(DTz) + lam*sum(abs(Dy-DDTz))  # Двойственная переменная, необходимая для оценки результата #2    

        # Оценка первичной цели

        pobj = min(pobj1,pobj2)  # Использование двойственной переменной двойственной задачи

        dobj = -0.5*DTz.T.dot(DTz) + Dy.T.dot(z)

        gap  =  pobj-dobj; # Параметр условия оптимальности

        gap_mass.append(gap)

        if (gap <= TOL): # Критерий остановки #1

            y = np.array([y])

            res = y.T-D.T.dot(z) # Вычисление результата фильтрации

            end_time_BOYD = time.time()

            all_time_BOYD_all.append(end_time_BOYD-start_time_BOYD)

            return(res) # Результат

            break;  

        if (i>=10): # Критерий остановки #2

            if (gap_mass[i-1]-gap_mass[i]<1e-7):

               y = np.array([y])

               res = y.T-D.T.dot(z) # Вычисление результата фильтрации

               end_time_BOYD = time.time()

               all_time_BOYD_all.append(end_time_BOYD-start_time_BOYD)

               return(res) # Результат

               break;

        if (step >= 0.2):

            t = max(2*m*MU/gap, 1.2*t) 

        # Вычисление шага Ньютона

        rz = DDTz - w

        b = (mu1/f1) + (mu2/f2)

        row_ind = []

        col_ind = []

        S_1 = [] # Создание разряженной матрицы

        for i in range(n-2):

            S_1.append(b[i][0])

            row_ind.append(i)

            col_ind.append(i)

        mat_coo = sparse.coo_matrix((S_1, (row_ind, col_ind)))

        S_2 = mat_coo.toarray()

        S = DDT - S_2

        r = -DDTz + Dy + (1/t)/f1 - (1/t)/f2

        dz =  np.linalg.lstsq(S,r,rcond=None)[0]

        dmu1  = -(mu1+((1/t)+dz*mu1)/f1) 

        dmu2 = -(mu2+((1/t)-dz*mu2)/f2)

        resDual = rz # Двойной остаток осуществимости

        resCent = np.vstack((-mu1*f1-1/t , -mu2*f2-1/t)) # Остаток центрирования

        residual = np.vstack((resDual, resCent)) # Остаток 

        # Поиск обратной линии

        negIdx1 = (dmu1 < 0)

        negIdx2 = (dmu2 < 0)

        step = 1;

        mu1_negIdx1 = []

        dmu1_negIdx1 = []

        mu2_negIdx2 = []

        dmu2_negIdx2 = []  

        index_stop = 0

        for i in range(len(negIdx1)): # Проверка булевы на TRUE для перерасчёта шага

            if negIdx1[i] == True:

                index_stop+=1

                mu1_negIdx1.append(mu1[i][0])

                dmu1_negIdx1.append(dmu1[i][0])

                if(i==len(negIdx1)-1):

                    step = min(step,0.99*np.min(np.array([mu1_negIdx1])*(-1)/dmu1_negIdx1))

        index_stop = 0

        for i in range(len(negIdx2)):

            if negIdx2[i] == True:

                index_stop+=1

                mu2_negIdx2.append(mu2[i][0])

                dmu2_negIdx2.append(dmu2[i][0])

                if(i==len(negIdx2)-1):

                    step = min(step,0.99*np.min(np.array([mu2_negIdx2])*(-1)/dmu2_negIdx2))

        for j in range(1,MAXLSITER):

            # Пересчёт прямых и двойственных переменных с учётом шага

            newz    =  z  + step*dz

            newmu1  =  mu1 + step*dmu1

            newmu2  =  mu2 + step*dmu2

            newf1   =  newz - lam

            newf2   = -newz - lam

            # Пересчёт остатков

            newResDual  = (DDT.dot(newz))  - Dy + newmu1 - newmu2

            newResCent  = np.vstack((-newmu1*newf1-1/t, -newmu2*newf2-1/t))

            newResidual = np.vstack((newResDual, newResCent))

            if ( max(max(newf1),max(newf2)) < 0 and np.linalg.norm(newResidual) <= (1-ALPHA*step)*np.linalg.norm(residual)):

                break;

            step = BETA*step

        # Обновление прямых и двойственных переменных

        z  = newz

        mu1 = newmu1 

        mu2 = newmu2 

        f1 = newf1 

        f2 = newf2



#####################################################################################################

# ФИЛЬТР ХОДРИКА-ПРЕСКОТТА

#####################################################################################################

all_time_HP_all = []

def HP_Filter(y,lam): #y,w

    start_time_HP = time.time()  # Определение времени начала работы фильтра

    y = np.atleast_2d(y)

    m,n  = y.shape # убедимся, что входные сигналы имеют правильную форму

    if m < n:

        y = y.T

        m = n

    a = np.array([lam, -4*lam, ((6*lam+1)/2.)]) # Вектор для расчета B (1)

    d = np.tile(a, (m,1)) # Расчёт матрицы Dx

    d[0,1]   = -2.*lam 

    d[m-2,1] = -2.*lam

    d[0,2]   = (1+lam)/2.

    d[m-1,2] = (1+lam)/2.

    d[1,2]   = (5*lam+1)/2.

    d[m-2,2] = (5*lam+1)/2.

    B = sparse.spdiags(d.T, [-2,-1,0], m, m) # Вычисление разряженной матрицы B из диагоналей m

    B = B+B.T # Вычисление новой матрицы B

    res = np.dot(la.inv(B.todense()),y) # Умножение обратной матрицы B на сигнал y

    end_time_HP = time.time() # Определение времени конца работы фильтра

    all_time_HP_all.append(end_time_HP-start_time_HP) # Расчет затраченного времени

    return res # Результат фильтрации
#####################################################################################################

print("1. Отрисовываем графики с исходными данными.")

#####################################################################################################

Graph_Display(force_y,'Исходные данные силы (F[Y])','Сила F[Y]','F[Y]')

Graph_Display(pose_x,'Исходные данные оси X','координата оси X','X')

Graph_Display(pose_y,'Исходные данные оси Y','координата оси Y','Y')

force_y_request = []

pose_x_request = []

pose_y_request = []
#####################################################################################################

# ОБРАБОТКА ДАННЫХ

#####################################################################################################

print("\n","2. Выделяем данные с условием рабочей области кобота.")

#####################################################################################################

for i in range(len(force_y)):

    if pose_x[i] > 0.2: 

        force_y_request.append(force_y[i]) # Усилия при X > 0.2

        pose_x_request.append(pose_x[i]) # Координата X при X > 0.2

        pose_y_request.append(pose_y[i]) # Координата Y при X > 0.2



print("Отрисовываем графики с выделенным условием.")

Graph_Display(force_y_request,'Данные силы (F[Y]) при X > 0.2','Усилие F[Y]','F[Y]')

Graph_Display(pose_x,'Данные оси X при X > 0.2','Координата X','X')

Graph_Display(pose_y,'Данные оси Y при > 0.2','Координата Y','Y')
#####################################################################################################

print("\n","3. Выделяем цикличность процесса и совмещаем результаты на одном графике, нормируя F[Y] для масштаба.")

#####################################################################################################

series=0 # Циклы процесса

iter_i=0 

iter_j=510

#Нормирование F[Y], относительно X

pose_x_min = np.min(pose_x) 

pose_x_max = np.max(pose_x)

force_y_min = np.min(force_y_request)

force_y_max = np.max(force_y_request)

scale_force_y = (pose_x_max-pose_x_min)/(force_y_max-force_y_min)

force_y_scale = [] 

for i in range(len(force_y_request)):

    force_y_scale.append(force_y_request[i]*scale_force_y)

force_y_series = [] 

force_y_scale_series = []

pose_series_x = []

pose_series_y = []

while (series<50): # Выделение данных через каждые 510 значений (цикличность процесса)

    if(iter_j<len(force_y_request)):

        force_y_series.append(force_y_request[iter_i:iter_j])

        force_y_scale_series.append(force_y_scale[iter_i:iter_j])

        pose_series_x.append(pose_x_request[iter_i:iter_j])

        pose_series_y.append(pose_y_request[iter_i:iter_j])

        iter_i+=510

        iter_j+=510

    series+=1

Graph_Display_Settings("Совмещённые графики (n=1) F[Y], X, Y по каждому циклу процесса",'F[Y], X, Y')

plt.plot(force_y_scale_series[0],color = "k", label = "F[Y]")

plt.plot(pose_series_x[0], color = "b", label = "X")

plt.plot(pose_series_y[0], color = "r", label = "Y")

plt.legend(loc="upper right") 

plt.show()

Graph_Display_Settings("Совмещённые графики (n=5) F[Y], X, Y по каждому циклу процесса",'F[Y], X, Y')

for i in range(5):

    plt.plot(force_y_scale_series[i],color = "k", label = "F[Y]")

    plt.plot(pose_series_x[i], color = "b", label = "X")

    plt.plot(pose_series_y[i], color = "r", label = "Y")

    plt.legend(loc="upper right") 

plt.show()
#####################################################################################################

print("\n",'4. Выделяем область, где X - прямая, а Y линейно возрастает.')

#####################################################################################################

series=0

iter_i=100 # Начало интервала

iter_j=300 # Конец интервала

force_y_series = [] 

force_y_scale_series = [] 

pose_series_x = []

pose_series_y = []

while (series<50):

    if(iter_j<len(force_y_request)):

        force_y_series.append(force_y_request[iter_i:iter_j]) # Интервалы силы

        force_y_scale_series.append(force_y_scale[iter_i:iter_j]) # Пронормированные интервалы силы

        pose_series_x.append(pose_x_request[iter_i:iter_j]) # Интервалы X

        pose_series_y.append(pose_y_request[iter_i:iter_j]) # Интервалы Y

        iter_i+=510 

        iter_j+=510

    series+=1

Graph_Display_Settings("Совмещённые графики (n=1) для F[Y], X, Y по каждому циклу процесса с учётом начальных условий",'F[Y], X, Y')

plt.plot(force_y_scale_series[0], label = "F[Y]")

plt.plot(pose_series_x[0], label = "X")

plt.plot(pose_series_y[0], label = "Y")

plt.legend(loc="upper right") 

plt.show()    

Graph_Display_Settings("Совмещённые графики (n=5) для F[Y], X, Y по каждому циклу процесса с учётом начальных условий",'F[Y], X, Y')    

for i in range(5):

    plt.plot(force_y_scale_series[i], label = "F[Y]")

    plt.plot(pose_series_x[i], label = "X")

    plt.plot(pose_series_y[i], label = "Y")

    plt.legend(loc="upper right") 

plt.show()

Graph_Display_Settings("Совмещённые графики (n=1) для F[Y] по каждому циклу процесса с учётом начальных условий",'F[Y]') 

plt.plot(force_y_series[0], label = "F[Y]")

plt.legend(loc="upper right") 

plt.show()

Graph_Display_Settings("Совмещённые графики (n=5) для F[Y] по каждому циклу процесса с учётом начальных условий",'F[Y]') 

for i in range(5):

    plt.plot(force_y_series[i], label = "F[Y]")

    plt.legend(loc="upper right") 

plt.show()
#####################################################################################################

# ФИЛЬТРАЦИЯ ДАННЫХ

#####################################################################################################

print("\n",'5. Используем фильтрацию Ходрика-Прескотта и Бойда. Производятся расчеты...')

#####################################################################################################

force_y_series_filtred_HP = []

force_y_series_filtred_BOYD = []

for i in range(len(force_y_series)):

    force_y_series_filtred_HP.append(HP_Filter(force_y_series[i],5.5))

    force_y_series_filtred_BOYD.append(Boyd_Filter(force_y_series[i],11.9))

Graph_Display(force_y_series_filtred_HP[0],'Отфильтрованные данные, фильтр Ходрика-Прескотта (n=1) для F[Y]','F[Y]','F[Y]')

Graph_Display(force_y_series_filtred_BOYD[0],'Отфильтрованные данные, фильтр Бойда (n=1) для F[Y]','F[Y]','F[Y]')

Graph_Display_Settings("Отфильтрованные данные, фильтр Ходрика-Прескотта (n=5) для F[Y]",'F[Y]')    

for i in range(5):

    plt.plot(force_y_series_filtred_HP[i], label = "F[Y]")

    plt.legend(loc="upper right") 

plt.show()

Graph_Display_Settings("Отфильтрованные данные фильтр Бойда (n=5) для F[Y]",'F[Y]')    

for i in range(5):

    plt.plot(force_y_series_filtred_BOYD[i], label = "F[Y]")

    plt.legend(loc="upper right") 

plt.show()
#####################################################################################################

# МЕТОД ПОИСКА КАСАНИЙ

#####################################################################################################

print("\n",'6. Выберем окно в n-точек и рассчитаем дисперсию, относительно каждой итерации процесса и окна.')

#####################################################################################################

# Выбираем окна, в которых находятся интервалы F[Y]

window_start = 0 # Начало окна

windows_end = 20 # Конец окна

window_force_y_series_filtred_HP = [] 

dispersion_window_force_HP = []

window_force_y_series_filtred_BOYD = [] 

dispersion_window_force_BOYD = []

for i in range(len(force_y_series_filtred_HP)):

    for j in range(len(force_y_series_filtred_HP[i])-19):

        window_force_y_series_filtred_HP.append(force_y_series_filtred_HP[i][window_start:windows_end]) #Все окна со всех данных (190 окон, 24 итерации)

        window_force_y_series_filtred_BOYD.append(force_y_series_filtred_BOYD[i][window_start:windows_end])

        window_start+=1

        windows_end+=1

    window_start = 0

    windows_end = 20

window_start_dispersion = 0 # Начало окна для дисперсии по каждой итерации процесса

window_end_dispersion = 180 # Конец окна для дисперсии по каждой итерации процесса

for i in range(len(window_force_y_series_filtred_HP)):

    dispersion_window_force_HP.append(np.var(window_force_y_series_filtred_HP[i]))

    dispersion_window_force_BOYD.append(np.var(window_force_y_series_filtred_BOYD[i])) # Дисперсия по каждому окну для всех итераций процесса

print("Дисперсия для одной итерации процесса, относительно каждого окна (Фильтр Ходрика-Прескотта)")

print(np.round(dispersion_window_force_HP[0:180],3), "\n")

print("Дисперсия для одной итерации процесса, относительно каждого окна (Фильтр Бойда)")

print(np.round(dispersion_window_force_BOYD[0:180],3))

dispersion_window_force_max_HP = []

dispersion_window_force_max_index_HP = []

dispersion_window_force_max_BOYD = []

dispersion_window_force_max_index_BOYD = []

print("\n","Отрисовываем графики с полученной дисперсией.")

Graph_Display(dispersion_window_force_HP[0:180],'Дисперсия фильтра Ходрика-Прескотта (n=1), относительно каждого окна','Дисперсия (D)','D')

Graph_Display(dispersion_window_force_BOYD[0:180],'Дисперсия фильтра Бойда (n=1), относительно каждого окна','Дисперсия (D)','D')

dispersion_window_force_x = 0

dispersion_window_force_y = 180

Graph_Display_Settings("Дисперсия фильтра Ходрика-Прескотта (n=5), относительно каждого окна",'Дисперсия (D)')    

for i in range(5):

    plt.plot(dispersion_window_force_HP[dispersion_window_force_x:dispersion_window_force_y], label = "D")

    plt.legend(loc="upper right") 

    dispersion_window_force_x+=181

    dispersion_window_force_y+=181

plt.show()

dispersion_window_force_x = 0

dispersion_window_force_y = 180

Graph_Display_Settings("Дисперсия фильтра Бойда (n=5), относительно каждого окна",'Дисперсия (D)')    

for i in range(5):

    plt.plot(dispersion_window_force_BOYD[dispersion_window_force_x:dispersion_window_force_y], label = "D")

    plt.legend(loc="upper right") 

    dispersion_window_force_x+=181

    dispersion_window_force_y+=181

plt.show()
#####################################################################################################

print("\n",'7. Рассчитаем максимальную дисперсию из всех окон для каждой итерации процесса:')

#####################################################################################################

window_dispersion_max_HP = []

window_dispersion_max_BOYD = []

for i in range(len(force_y_series_filtred_HP)):

    dispersion_window_force_max_HP.append(np.max(dispersion_window_force_HP[window_start_dispersion:window_end_dispersion])) # Максимальное значение дисперсии из всех окон каждой итерации процесса

    dispersion_window_force_max_index_HP.append(dispersion_window_force_HP.index(np.max(dispersion_window_force_HP[window_start_dispersion:window_end_dispersion]))) # Индекс окна с максимальной дисперсией по каждой итерации процесса

    window_dispersion_max_HP.append(window_force_y_series_filtred_HP[dispersion_window_force_max_index_HP[i]])

    dispersion_window_force_max_BOYD.append(np.max(dispersion_window_force_BOYD[window_start_dispersion:window_end_dispersion])) # Максимальное значение дисперсии из всех окон каждой итерации процесса

    dispersion_window_force_max_index_BOYD.append(dispersion_window_force_BOYD.index(np.max(dispersion_window_force_BOYD[window_start_dispersion:window_end_dispersion]))) # Индекс окна с максимальной дисперсией по каждой итерации процесса

    window_dispersion_max_BOYD.append(window_force_y_series_filtred_BOYD[dispersion_window_force_max_index_BOYD[i]])

    window_start_dispersion+=181

    window_end_dispersion+=181

print("Максимальная дисперсия фильтра Ходрика-Прескотта:")

print(np.round(dispersion_window_force_max_HP,3), "\n")

print("Максимальная дисперсия фильтра Бойда")

print(np.round(dispersion_window_force_max_BOYD,3))
#####################################################################################################

print("\n",'8. Выведем окна c максимальной дисперсией.')

#####################################################################################################

Graph_Display_Settings("Окно для F[Y] фильтра Ходрика-Прескотта (n=1) с максимальной дисперсией",'F[Y]')    

plt.plot(window_force_y_series_filtred_HP[dispersion_window_force_max_index_HP[0]], label = "F[Y]")

plt.legend(loc="upper right") 

plt.show()

Graph_Display_Settings("Окна для F[Y] фильтра Ходрика-Прескотта (n=5) с максимальной дисперсией",'F[Y]')    

for i in range(5):

    plt.plot(window_force_y_series_filtred_HP[dispersion_window_force_max_index_HP[i]], label = "F[Y]")

    plt.legend(loc="upper right") 

plt.show()

Graph_Display_Settings("Окно для F[Y] фильтра Бойда (n=1) с максимальной дисперсией",'F[Y]')    

plt.plot(window_force_y_series_filtred_BOYD[dispersion_window_force_max_index_BOYD[0]], label = "F[Y]")

plt.legend(loc="upper right") 

plt.show()

Graph_Display_Settings("Окна для F[Y] фильтра Бойда (n=5) с максимальной дисперсией",'F[Y]')    

for i in range(5):

    plt.plot(window_force_y_series_filtred_BOYD[dispersion_window_force_max_index_BOYD[i]], label = "F[Y]")

    plt.legend(loc="upper right") 

plt.show()
#####################################################################################################

print("\n","9. Определим минимальную разницу (sigma) дисперсий между максимальной дисперсией окна и средней по окнам, лежащим до максимальной.")

#####################################################################################################

dispersion_window_force_difference_HP = []

dispersion_window_force_not_max_HP = []

dispersion_window_force_difference_BOYD = []

dispersion_window_force_not_max_BOYD = []

iter_disp = 0

for i in range(len(dispersion_window_force_max_HP)):

    dispersion_window_force_not_max_HP.append(dispersion_window_force_HP[iter_disp:dispersion_window_force_max_index_HP[i]-1])

    dispersion_window_force_difference_HP.append(dispersion_window_force_max_HP[i]-np.mean(dispersion_window_force_not_max_HP[i]))

    dispersion_window_force_not_max_BOYD.append(dispersion_window_force_BOYD[iter_disp:dispersion_window_force_max_index_BOYD[i]-1])

    dispersion_window_force_difference_BOYD.append(dispersion_window_force_max_BOYD[i]-np.mean(dispersion_window_force_not_max_BOYD[i]))

    iter_disp+=180

dispersion_window_force_mean_difference_HP = np.min(dispersion_window_force_difference_HP)

dispersion_window_force_mean_difference_BOYD = np.min(dispersion_window_force_difference_BOYD)

print("Sigma фильтрации Ходрика-Прескотта) =", round(dispersion_window_force_mean_difference_HP,3))

print("Sigma фильтрации Бойда) =", round(dispersion_window_force_mean_difference_BOYD,3))
#####################################################################################################

print("\n","10. Cравним, насколько в среднем отличаются выбранные окна при использовании 2-х разных фильтров.")

#####################################################################################################

difference_filters = []

for i in range(len(dispersion_window_force_max_index_HP)):

    difference_filters.append(abs(dispersion_window_force_max_index_HP[i]-dispersion_window_force_max_index_BOYD[i])) 

difference_filters = np.mean(difference_filters)

print("10.1 Средняя разница между выбранными окнами фильтра Ходрика-Прескотта и фильтра Бойда составляет ", difference_filters, "окна")

print("10.2 Среднее время, затрачиваемое для фильтрации одного окна по Ходрику-Прескотту = ", round(np.mean(all_time_HP_all),5),'секунды')

print("10.2 Среднее время, затрачиваемое для фильтрации одного окна по Бойду = ", round(np.mean(all_time_BOYD_all),5),'секунды')

if (np.mean(all_time_HP_all)<np.mean(all_time_BOYD_all)):

    print("Несмотря на то, что фильтры дают схожий результат, фильтр Ходрика-Прескотта оказался значительно быстрее, чем фильтр Бойда.") 

else:

    print("Несмотря на то, что фильтры дают схожий результат, фильтр Бойда оказался значительно быстрее, чем фильтр Ходрика-Прескотта.")
#####################################################################################################

# ВЫВОД РЕЗУЛЬТАТА В ФАЙЛЫ .CSV

#####################################################################################################

data_windows = {

    "windows_hp": np.ravel(window_dispersion_max_HP),

    "windows_boyd": np.ravel(window_dispersion_max_BOYD)

}

data_dispersion = {

"windows_dispersion_max_hp":dispersion_window_force_max_HP,

"windows_dispersion_max_boyd": dispersion_window_force_max_BOYD

}

data_time = {

    "average_time_hp": all_time_HP_all,

    "average_time_boyd": all_time_BOYD_all

}

data_sigma = {

"sigma_hp":np.ravel(dispersion_window_force_mean_difference_HP),

"sigma_boyd":np.ravel(dispersion_window_force_mean_difference_BOYD)

}

dataframe = pd.DataFrame(data_windows)

dataframe.to_csv("collision.csv")

dataframe = pd.DataFrame(data_dispersion)

dataframe.to_csv("dispersion.csv")

dataframe = pd.DataFrame(data_time)

dataframe.to_csv("time.csv")

dataframe = pd.DataFrame(data_sigma)

dataframe.to_csv("sigma.csv")