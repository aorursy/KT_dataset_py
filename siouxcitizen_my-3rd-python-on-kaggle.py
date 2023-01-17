#配列：リスト　数値のみ表示

print("数値の配列データのみ表示")

list1 = [1, 2, 3, 4, 5]

print(list1)



list2 = [1, 2, 4, 8, 16, 32, 64, 128, 256]

print(list2)



list1 = list1 + [6, 7, 8, 9, 10]

print(list1)



list1 = list1 + list2

print(list1)



print("\n")



#配列：リスト　数値と文字列連結して表示

print("数値の配列データと文字列を連結して表示")

list1 = [1, 2, 3, 4, 5]

print("list1 は  " + str(list1))



list2 = [1, 2, 4, 8, 16, 32, 64, 128, 256]

print("list2 は  " + str(list2))



list1 = list1 + [6, 7, 8, 9, 10]

print("list1 + [6, 7, 8, 9, 10] は  " + str(list1))



list1 = list1 + list2

print("list1 + list2 は  " + str(list1))



print("\n")



#配列：リスト　文字列の表示

print("文字列の配列データのみ表示")

list3 = ["あああ", "いいい", "ううう"]

print(list3)

list4 = ["かかか", "ききき", "くくく"]

print(list4)

list3 = list3 + ["eee", "ooo"]

print(list3)

list4 = list4 + ["kekeke", "kokoko"]

print(list4)

list3 = list3 + list4

print(list3)



print("\n")



#配列：リスト　文字列と文字列連結して表示

print("文字列の配列データと文字列を連結して表示")

list3 = ["あああ", "いいい", "ううう"]

print("list3 は " + str(list3))



list4 = ["かかか", "ききき", "くくく"]

print("list4 は " + str(list4))



list3 = list3 + ["eee", "ooo"]

print("list3 + [\"eee\", \"ooo\"] は " + str(list3))



list4 = list4 + ["kekeke", "kokoko"]

print("list4 + [\"kekeke\", \"kokoko\"] は " + str(list4))
