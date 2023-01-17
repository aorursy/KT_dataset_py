import pip
pip.main(['install', 'pysimplegui'])
import PySimpleGUI as sg

layout = [ [sg.Text('Ingrese un Número'), sg.Input()],
           [sg.OK('Aceptar')],
         ]

window = sg.Window('Serie de Fibonacci', layout)
event, values = window.Read()
sg.Popup('El Valor Ingresado es:', values[0])
window.close()
def fib(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        return fib(n-1) + [1] 
    result = fib(n-1)
    return result + [result[-1] + result[-2]] 
    
fib(10)
layout = [ [sg.Text('Ingrese un Número'), sg.Input()],
           [sg.OK('Aceptar')],
         ]

window = sg.Window('Serie de Fibonacci', layout)
event, values = window.Read()

try:
    n = int(values[0])
    sg.Popup(f'Los primeros {n} dígitos de la serie de Fibonacci son:', ' '.join(map(str, fib(n)))) 
except ValueError as e:
    sg.Popup(f'El Valor Ingresado es errado: {e}')

window.close()
layout = [ [sg.Text('Ingrese un Número'), sg.Input()],
           [sg.OK('Aceptar'),sg.Cancel('Cancelar')],
         ]

window = sg.Window('Serie de Fibonacci', layout)
event, values = window.Read()

print (f"El evento accionado fue {event}")
    
if event is not None and event != 'Cancelar':
    try:
        n = int(values[0])
        sg.Popup(f'Los primeros {n} dígitos de la serie de Fibonacci son:', ' '.join(map(str, fib(n)))) 
    except Exception as e:
        sg.Popup(f'El Valor Ingresado es errado: {e}')

window.close()

