'''
開始練習變數設定, 變數只能以大小寫英文字母與 _ 開頭, 隨後可以接上大小寫字母或底線以及阿拉伯數字
變數其實也可以利用中文命名, 但是並不建議, 因為隨後的程式可能必須與非中文語系的其他學員或工程師進行溝通或協同編輯
'''
# 選定 studNumber 作為變數, 而且與字串 "50703199" 對應
studNumber = "50703199"
# 利用 print() 函式列出 studNumber 變數的資料型別
#print(type(studNumber))
# 可以利用 help() 函式列出 print 函式的說明
#print(help(print))
'''
Help on built-in function print in module builtins:

print(...)
    print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)
    
    Prints the values to a stream, or to sys.stdout by default.
    Optional keyword arguments:
    file:  a file-like object (stream); defaults to the current sys.stdout.
    sep:   string inserted between values, default a space.
    end:   string appended after the last value, default a newline.
    flush: whether to forcibly flush the stream.

None
'''
# 接下來可以試著了解 print() 這個內建函式的更進階用法
# sep seperate, \n, next line
for i in range(5):
    print(studNumber, "test", "w7", sep=":")