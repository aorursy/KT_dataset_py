paragraph = """Bir düzenli ifade ile tanımlanan bir dizi özel karakterden oluşan bir kalıp sayesinde bir metin içerisinde çok karmaşık ya da çok basit aramalar +

değiştirmeler yapabiliriz. Burada bazı kurallar ve özel bir söz dizimi vardır. 

Regex, düzenli ifadeler için kullanılan bir kısaltmadır, REGular EXpressions."""
# düzenli kelimesini göstermek istediğimizi varsayalım veya diğer bir deyişle 4. ile 11. karakter arasını... not= 4 dahil 1 haric

# Let's say we want to show the word regular, or in other words between 4 and 11 characters ... note: Including 4 except 11



paragraph[4:11]

kelime = re.search("kurallar",paragraph) #match=aranan değer, span=nere olduğunu 186 ve 194. karakter arasındadır.
kelime.start(),kelime.end()

control = re.search("düzenli",paragraph)

control.endpos  #304

# paragraph'a düzenli kelimesinin nerede geçtiğini sorup cevabını control'e atadım. 

# control'den yararlanarak ilgili pos'un kaç karakterden oluştuğunu sordum ve bana 304 cevabını verdi

# Doğrulamak için aşağıda paragraph'ın 304 karakterini yansıtmasını istedim. Bu şekilde doğrulamış oldum.

paragraph[:304]
re.findall("düzenli",paragraph) # ['düzenli','düzenli']

len(re.findall("düzenli",paragraph))
paragraph = "la ilahe illallah"
len(re.findall(".la",paragraph)) # paragraph içerisinde "la" ikilsinin tekrarlanma sayısını istedik
len(re.findall("ah",paragraph))
paragraph = """

            Mail Adreslerim: 

            karagoz29kursat@gmail.com, 

            karagoz29kursat@@@gmail.com, 

            karagoz29kursat@gmail.commm, 

            karagoz29kursat@@@@@@@@@@@gmail.com,

            @@@gggg@@@gggmail,

            @g@gmail,

            --@--@--@gmail.com.

"""
# "@" ifadesinin tekrarlanması önemsizdir.

# "@" chracters count isn't important

len(re.findall("@*gmail.com",paragraph))  #   
re.findall("@g+mail",paragraph)
re.findall("@g?mail",paragraph)
paragraph = """

            Ceren       #1      

            Seren       #2

            Geren       #3

            Neren       #4

            Hikmet

            Ayşe

            Birkeren    #5

            İkiKeren

            Yahya,

            uuuuuuutku,

            uuuuuuufuk,

            Keferen    #6    

"""
re.findall("[CSGNkf]eren",paragraph)

# Öncelikle eren kelimesinin geçtiği yerleri bulur ve daha sonra:

# Dizi içerisine yazılan karakterleri tek tek , dizilişi ile yan yana ve dizi bütünü ile tüm olasılıları başa getirip arar.
re.findall("u{5}fuk",paragraph)
metin = "Elmalar kırmızı muzlar sarıdır"



# (.*) Grubu ile herhangi bir karakteri bir veya daha fazla kez tekrar edebileceğini söyledik ta ki "lar" karakter bütününü görene kadar.

# Daha sonra bir " " (boşluk) arar.

# Boşluğu bulduktan sonra (.*?) ile " herhangi bir karakterin bir veya daha fala geçtiği bir tanım var " bu ifadeyi bir kez bulmalı ya da hiç bulmamalı.

# Daha sonra tekrar boşluk arar.

# son olarak .* tanımı il eherhangi bir karakteri defalarca kez eşleştirebilirsin diyoruz.

eslesme = re.match( '(.*)lar (.*?) .*', metin, re.I)

if eslesme:

   print(eslesme.groups())                           #Elma,kırmızı

   #print("eslesme.group() : ", eslesme.group())     #ifadenin tamamı

   #print("eslesme.group(1) : ", eslesme.group(1))   #Elma

   #print("eslesme.group(2) : ", eslesme.group(2))   #kırmızı

else:

   print("Eşleşme yok")
metin = "Elmalar kırmızı muzlar sarıdır"

 

eslesme = re.match( '(muz)', metin, re.I)

arama = re.search( '(muz)', metin, re.I)

 

if eslesme:

   print(eslesme.groups())

else:

   print("Eşleşme yok")

   

if arama:

   print(arama.groups())

else:

   print("Arama bulunamadı")
metin = "Elmalar kırmızı muzlar sarıdır"

 

degisim = re.sub( 'elma', 'kiraz', metin, flags=re.I)    # elma ==> kiraz

 

print(degisim)

 

# kirazlar kırmızı muzlar sarıdır