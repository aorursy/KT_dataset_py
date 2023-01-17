# İlk olarak import edilir. 

# Kişiye bağlı kullanımdır ama literatürde np olarak adlandırılır. 



import numpy as np 
veri = [[1,2,3],[4,5,6],[7,8,9]]



a = np.array(veri) # Eldeki veriyi array formatına çevirir.



a
print(a[0])    # 0. indexteki veriyi alır.

print(a[1,2])  # 1. indexin 2. verisini alır.
f = np.arange(0, 500, 3)



index_deger = [2, 34, 23, 40]



f[index_deger]
print(a.ndim)

print(a.shape)

print(a.size)

print(a.dtype)
print(np.arange(2,30)) 

print(np.arange(2,30,3)) # Verilen değer kadar atlayarak da depolar.
print(np.zeros(30)) # Belirtilen miktar kadar 0 oluşturur.

print(np.ones(20))  # Belirtilen miktar kadar 1 oluşturur.
np.zeros((10,5)) # Çift parantez ile boyut düzenlendi.
np.full((3,4),8) # Belirli değerde matris oluşturur.
np.eye(5) # 5x5'lik birim matris oluşturur.
np.linspace(2,50,10)
print(np.random.randint(0,100))   # 0 ile 100 arasında rastgele bir sayı üretir.

print(np.random.randint(100))     # Başa sıfır yazılmadığında da üsttekiyle aynı işlevle çalışır.

print(np.random.randint(0,100,4)) # Diğer fonksiyonlarda olduğu argümanla da sayı üretir.

print(np.random.rand(5))          # 0 ile 1 arasında 5 sayı üretir.
sekil = np.arange(30)



print(sekil)              # Tek boyutlu yani vektörel.

print(sekil.reshape(5,6)) # Çok boyutlu yani matris.
x = np.array([1,2,3])

y = np.array([4,5,6])

z = [7,8,9]



np.concatenate([x,y,z]) # Değerleri birleştirir.
v = np.array([34, 4, 1, 9, 10])



print(v)



print(np.sort(v))    # NumPy'daki metotla sıralar, array orjinalinde değişiklik yapmaz. v.sort() olsaydı kalıcı değişirdi.

print(np.argsort(v)) # Sıralama sonrası değerlerin eski index değerlerini, yani nerede olduklarını döndürür.
stats = np.arange(13,50)



print(stats)

print(np.cumsum(stats)) # Kümülatif toplam.

print(np.sum(stats))    # Tümünün toplamı.

print(np.min(stats))    # Mininum değer.

print(np.max(stats))    # Maksimum değer.

print(np.std(stats))    # Standart sapması.

print(np.var(stats))    # Varyansı.

print(np.mean(stats))   # Ortalaması.

print(np.median(stats)) # Ortanca değeri.
print(np.argmax(stats)) # En büyük değerin indexini verir.

print(np.argmin(stats)) # En küçük değerin indexini verir.
dilim = np.arange(0,100)



dilim[2:7]
ikinci_dilim = np.arange(0,30).reshape(5,6)



print(ikinci_dilim)

print(ikinci_dilim[3,:])     # Satırın tamamını alır.

print(ikinci_dilim[:,3])     # Sütunun tamamını alır.

print(ikinci_dilim[0:2,1:4]) # Matrislerde virgül kesişim kümesi gibi çalışır.



ikinci_dilim[1:4,0:3] = 99   # Değerler için atama işlemi de yapılabilir.



print(ikinci_dilim)
print(ikinci_dilim)



print(ikinci_dilim[0:2,1:4])



alt = ikinci_dilim[0:2,1:4].copy()



alt[0:1,1:] = 8888 # Alt arrayde değişiklik yapıyoruz.



print(alt)



print(ikinci_dilim) # Ama yaptığımız değişiklik ana arrayi etkilemiyor.
ikinci_dilim > 20
a = np.array([1,2,3,4,5])

b = np.array([6,7,8,9,10])



print(a)

print(a-1)

print(a+1)

print(a*3)

print(a**2)

print(a/2)



print(a*b)
print(np.add(a,2))      # Toplama

print(np.subtract(a,2)) # Çıkarma

print(np.multiply(a,2)) # Çarpma

print(np.divide(a,2))   # Bölme

print(np.power(a,2))    # Üs