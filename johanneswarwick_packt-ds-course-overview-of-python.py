var1 = 8
var2 = 160.88
var1 + var2
var3 = 'Hello, '
var4 = 'World'
print(var3)
print(var4)
print(f"Text: {var3} {var4}!")
var3 + var4
var5 = ['I', 'love', 'data', 'science']
print(var5)
var6 = ['Packt', 15019, 2020, 'Data Science']
print(var6)
print(var6[0])
print(var6[2])
print(var6[0:3])
for item in var6:
  print(item)
var6.append('Python')
print(var6)
var6.remove(15019)
print(var6)
var7 = {'Topic': 'Data Science', 'Language': 'Python'}
print(var7)
var7['Language']
var7.keys()
var7.values()
for key, value in var7.items():
  print(key)
  print(value)
var7['Publisher'] = 'Packt'
print(var7)
del var7['Publisher']
print(var7)