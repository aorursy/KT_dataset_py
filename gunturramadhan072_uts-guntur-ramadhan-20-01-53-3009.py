nl = [0] * (11)

jum = 0
for lp in range(0, 10 + 1, 1):
    nl[lp] = int(input())
    jum = nl[lp] + jum
rata = jum / 11
print("rata anda:" + str(rata))
jmlh = 0
for lp in range(0, 10 + 1, 1):
    x = (nl[lp] - rata) ** 2
    jmlh = nl[lp] + jmlh
b = jmlh / (11 - 1)
hasildev = b ** 0.5
print("hasil deviasi anda:" + str(hasildev))
print("banyaknya data:" + str(lp))
for lp in range(0, 10 + 1, 1):
    print("data yang anda masukan" + str(nl[lp]))
