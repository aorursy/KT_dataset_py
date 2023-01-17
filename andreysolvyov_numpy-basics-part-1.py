import numpy as np
print(np.__version__)
o = np.zeros(10)

o[4] = 1

o[-3:] = 2

print(o)
viborka = np.random.uniform(150, 200, 100)

mean = np.mean(viborka)

standartnoe_otklonenie = np.std(viborka)

k = 1

percent = len(viborka[(mean - k*standartnoe_otklonenie < viborka) & (viborka < mean + k*standartnoe_otklonenie)])

print()
def close(x, y, eps):

    return np.abs(x - y) < eps

setka = np.linspace(0, np.pi/2, 20)

cos = np.cos(setka)

sinp_2_a = np.sin(np.pi/2 - setka)

eps = float(input())

print(close(cos, sinp_2_a, eps))
matr = np.random.normal(5, 2.5, (10, 5))

sr5 = np.mean(matr[4, :])

me3 = np.median(matr[:, 2])

sumr = np.sum(matr, axis = 1)

sumo = np.sum(matr, axis = 0)

suma = np.sum(matr)

det = np.linalg.det(matr[:3, :3])

kov = np.sum(matr, axis = 1)[np.sum(matr, axis = 1) < np.mean(np.sum(matr, axis = 1))].size

pra = matr[matr[:, 0] > matr[:, -1]].shape

rank = np.linalg.matrix_rank(matr[np.sum(matr, axis = 1) > np.mean(np.sum(matr, axis = 1))])
df = np.loadtxt('../input/cardio_train.csv', delimiter=';', skiprows=1)

shape = df.shape

df[:, 1] = np.floor(df[:, 1]/365.25)
co50 = df[df[:, 1] > 50].shape[0]

mean = np.mean(df[:, 3][df[:, 2] == 2]) > np.mean(df[:, 3][df[:, 2] == 1])

weightma = np.max(df[:, 4])

weightmi = np.min(df[:, 4])

couwe = df[df == np.max(df[:, 4])].size

couns = df[df[:, -4] == 0][:, -1][df[df[:, -4] == 0][:, -1] == 1].size