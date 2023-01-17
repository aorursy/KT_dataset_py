
pin = [(1234), (4567)]
account_number = [(11111), (22222), 'l']



def pin_change():
        p_pin = int(input('\n enter parmanent pin: '))
        if p_pin in pin:    
            
                pin_change1 = input(' \n you want to change pin  y/n \n ').lower()
        
                if pin_change1 == 'y':                                                   
                                     
                            pre_pin = int(input('\n enter previous pin: '))                            
                            if pre_pin == p_pin:                  
                                                                        
                                    change_pin = int(input('\n enter new four digit pin:   '))
                    
                                    if change_pin in  range(1000,10000):
                                        
                                            pin[0] = change_pin                                                                       
                                            print('\n pin change succesfully: ')
                                            print(pin)
                                            if more_oper == 'y':                        
                        
                                                    enter_pin()
                            
                                            else:
                                                    print('\n thnx')
            
                                            
                            
                                    else:
                                    
                                            print('\n enter 4 digit no: ')
                                            pin_change()
                                        
                            else:
                                        print('\n previous pin is wrong \n please enter right pin: ')
                                        pin_change()
                                    
                    
        
                    
                elif pin_change1 == 'n':
            
                           
                            print('\n thnx to visit:  ')        
                else: 
                            
                            print('\n please select y / n : ')                
                            pin_change()
                            
        else: 
            print('\n please enter valid: ')
            pin_change()

            
            
def pass_book():
        p_pin = int(input('\n enter parmanent pin: '))
        if p_pin in pin:  
          
                pass_book = input('\n please select \n 1. new pass book  \n 2.updated pass book  \n ')
            
                if pass_book == '1':
                
                    input('\n enter your account number:  ')
                    input('\n enter your phone no: ')
                    print('\n bank send your password at your address ')
                    more_oper = input('\n for more operation y/n:  ')
                    if more_oper == 'y':                        
                        
                            enter_pin()
                            
                    else:
                            print('\n thnx')
            
                elif pass_book == '2':
                
                    print('\n please insert your passbook in machine \n your passbook is updated  ')
                    if more_oper == 'y':                        
                        
                            enter_pin()
                            
                    else:
                            print('\n thnx')
            
            
                else:
                    print ('\n thanx to visit')
                
        else:
                print('\n enter valid pin')
                pass_book()

                
        
def current_balance(): 
        p_pin = int(input('enter parmanent pin: '))
        if p_pin in pin:       

                b_pin = int(input('\n enter your pin:  \n '))
            
                if b_pin in range(999,9999):
                
                    print('\n current balance is : 1000')
                    if more_oper == 'y':                        
                        
                            enter_pin()
                            
                    else:
                            print('\n thnx')
            
            
                else:
                    print('\n enter 4 digit number: \n ')
                    current_balance()
                
        else:
                print('\n enter valid pin')
                current_balance()
                
def mini_statement():
    p_pin = int(input('enter parmanent pin: '))
    if p_pin in pin:
    
            mini = input(' \n enter the duration: ')
            print(' \n mini statement is :   ', mini)
            more_oper = input('\n  more operation y/n: ')
            if more_oper == 'y':                        
                        
                            enter_pin()
                            
            else:
                            print('thnx')
            
            
    else:
            print('enter valid pin')
            mini_statement()
                
def cash_withdrawl():
        p_pin = int(input('enter parmanent pin: '))
        if p_pin in pin:
    
                    acc_type = input('enter type of account \n  current account: \n saving account: \n  ')
        
                    if acc_type == 'saving':
                
                        c_w_pin = input('enter pin number: ')
                        amount = input('please enter amount: ')
                        print(' please collect {} rupees '.format(amount))
                        more_oper = input(' more operation y/n: \n y --> withdrawl from current account \n n --> main menu')
                        if more_oper == 'y':                        
                        
                            cash_withdrwal()
                            
                        elif more_oper == 'n':
                                enter_pin()
                                
                        else:
                            
                                print('thnx')
       
                    elif  acc_type == 'current':
                        input('institiution id no: ')
                        c_w_pin = input('enter pin number: ')
                        print('amount should be greater than 50 tousand rupees: ')
                        amount = input('please enter amount: ')
                        print(' please collect {} rupees '.format(amount))
                        more_oper = input(' more operation y/n: \n y --> withdrawl from saving account \n n --> main menu')
                        if more_oper == 'y':                        
                        
                            cash_withdrwal()
                            
                        elif more_oper == 'n':
                                enter_pin()
                                
                        else:
                            
                                print('thnx')
                    
                    else: 
                        print('thanks')
                    
        else:
                print('enter valid pin')
                cash_withdrwal()
        
def fund_transfer():
        p_pin = int(input('enter parmanent pin: '))
        if p_pin in pin:
            
                f_pin = int(input('enter your pin: '))
        
                if f_pin in pin:
                
                    details = input(' \n enter amount '); input('\n IFSC code: '); input('\n enter account number')
                    print('\n ',details, ' amount is transfer')
                    more_oper = input(' \n more operation y/n: ')
                    if more_oper == 'y':
                                
                                enter_pin()
                                
                    else:
                            
                                print('thnx')
                
                else:
                    print('enter valid pin no: \n ')
                    fund_transfer()
                
        else:
                print('enter valid pin')
                fund_transfer()

        
def enter_pin(): 
    enter_pin1 = int(input('enter 4 digit pin'))
    if enter_pin1 in pin:
        
        print('which type of work you want to do: press one digit  \n 1.Services \n 2.balance enqiry: \n 3.Fund Transfer:   \n ')
        task  = input()
        if task == '1':
            
                print('which type of services yu want \n 1. Pin Change:  \n 2. passbook: \n')
                task1 = input()
                
                if task1 == '1':
                    pin_change()
                    
                elif task1 == '2':
                    pass_book()
                    
                else:
                    print(' thanks ')
            
        elif task == '2':
            
                print('balance enquiry  \n 1. current balance :  \n 2. mini statement: \n')
                task3 = input()
                if task3 == '1':
                    
                    current_balance()
                elif task3 == '2':
                    mini_statement()
                    
                else:
                    print('thanks to visit')
                    
        elif task == '3':
            
                    print('enter details: ')
                    fund_transfer()
            
        else: 
            print('thanks ')
            
    else:
            enter_pin()
            
        
enter_pin()    

        
