

class ourlist():



    def __init__(self,name,last_name,id_number,salary,language):

        self.name=name

        self.last_name=last_name

        self.id_number=id_number

        self.salary=salary

        self.language=language



    def show_info(self):

        print("""

        This our company's 1st sgt's info



        name : {}



        last_name : {}



        id_number : {}



        salary : {}



        language : {}.



        """.format(self.name,self.last_name,self.id_number,self.salary,self.language))



    def add_salary(self,add_money):

        print("Salary is going up")



        self.salary += add_money



    def add_language(self,new_lan):

        print('Language list extends')

        self.language.append(new_lan)



result=ourlist('Rauf','Safarov',123456,6800,['Python','SQL'])

result.show_info()



result.add_salary(1200)

result.show_info()



result.add_language('Tabealu')

result.show_info()
