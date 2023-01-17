class Camera:
    def take_picture(self):
        print("Picture has taken.")
class Lantern:
    def open_light(self):
        print("Light is opened.")

    def close_light(self):
        print("Light is closed.")
class Phone:
    def call_someone(self, phone):
        if phone:
            print("{} is calling".format(phone))
        else:
            raise ValueError("Phone must be provided")
            
"""
__name__ is a special variable for Python. It will be main if and only if this file is run directly.
Below statement will not work, if Phone class is imported to other classes.
"""
if __name__ == "__main__":
    phone = Phone()
    phone.call_someone("+5329998877")
class AndroidPhone(Camera, Lantern, Phone):
    pass
mi6 = AndroidPhone()  # Attention! if block at Phone class does not work.
mi6.call_someone("+905371112233")
mi6.open_light()
mi6.take_picture()
mi6.close_light()